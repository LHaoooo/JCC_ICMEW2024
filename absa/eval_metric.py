# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run MABSA on Twitter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import collections

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import f1_score, accuracy_score

from squad.squad_evaluate import exact_match_score
from absa.utils import read_absa_data, convert_absa_data, convert_examples_to_features, \
    RawFinalResult, wrapped_get_final_text, id_to_label

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
    from xml.sax.saxutils import escape
except:
    sys.exit('Some package is missing... Perhaps <re>?')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def read_train_data(args, tokenizer, logger):
    if args.debug:
        args.train_batch_size = 8

    train_path = os.path.join(args.data_dir, args.train_file)
    train_set = read_absa_data(train_path)
    train_examples = convert_absa_data(dataset=train_set, verbose_logging=args.verbose_logging)
    train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length,
                                                  args.verbose_logging, logger)

    num_train_steps = int(
        len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    logger.info("Num orig examples = %d", len(train_examples))
    logger.info("Num split features = %d", len(train_features))
    logger.info("Batch size = %d", args.train_batch_size)
    logger.info("Num steps = %d", num_train_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_span_starts = torch.tensor([f.start_indexes for f in train_features], dtype=torch.long)
    all_span_ends = torch.tensor([f.end_indexes for f in train_features], dtype=torch.long)
    all_labels = torch.tensor([f.polarity_labels for f in train_features], dtype=torch.long)
    all_label_masks = torch.tensor([f.label_masks for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_span_starts, all_span_ends,
                               all_labels, all_label_masks)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    return train_dataloader, num_train_steps

def read_eval_data(args, tokenizer, logger):
    if args.debug:
        args.predict_batch_size = 8

    eval_path = os.path.join(args.data_dir, args.predict_file)
    eval_set = read_absa_data(eval_path)
    eval_examples = convert_absa_data(dataset=eval_set, verbose_logging=args.verbose_logging)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,
                                                 args.verbose_logging, logger)

    logger.info("Num orig examples = %d", len(eval_examples))
    logger.info("Num split features = %d", len(eval_features))
    logger.info("Batch size = %d", args.predict_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_span_starts = torch.tensor([f.start_indexes for f in eval_features], dtype=torch.long)
    all_span_ends = torch.tensor([f.end_indexes for f in eval_features], dtype=torch.long)
    all_label_masks = torch.tensor([f.label_masks for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_span_starts, all_span_ends,
                              all_label_masks, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)
    return eval_examples, eval_features, eval_dataloader

def metric_max_over_ground_truths(metric_fn, term, polarity, gold_terms, gold_polarities):
    hit = 0
    for gold_term, gold_polarity in zip(gold_terms, gold_polarities):
        score = metric_fn(term, gold_term)
        if score and polarity == gold_polarity:
            hit = 1
    return hit

def metric_mate(metric_fn, term, gold_terms):
    hit = 0
    for gold_term in gold_terms:
        score = metric_fn(term, gold_term)
        if score:
            hit = 1
    return hit

def masc_metric(metric_fn, term, polarity, gold_terms, gold_polarities):
    classygood = 0
    for gold_term, gold_polarity in zip(gold_terms, gold_polarities):
        if metric_fn(term, gold_term) and polarity == gold_polarity :
            classygood = 1
    return classygood

def eval_absa(all_examples, all_features, all_results, do_lower_case, verbose_logging, logger):
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_nbest_json = collections.OrderedDict()
    common, relevant, retrieved = 0., 0., 0.
    # enti= 0.
    
    for (feature_index, feature) in enumerate(all_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        pred_terms = []
        pred_polarities = []
        for start_index, end_index, cls_pred, span_mask in \
                zip(result.start_indexes, result.end_indexes, result.cls_pred, result.span_masks):
            if span_mask:
                final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                    do_lower_case, verbose_logging, logger)
                pred_terms.append(final_text)
                pred_polarities.append(id_to_label[cls_pred])

        prediction = {'pred_terms': pred_terms, 'pred_polarities': pred_polarities,'gold_terms':example.term_texts,'gold_polarites':example.polarities}
        all_nbest_json[example.example_id] = prediction

        for term, polarity in zip(pred_terms, pred_polarities):
            common+= metric_max_over_ground_truths(exact_match_score, term, polarity, example.term_texts, example.polarities)
            
        retrieved += len(pred_terms)
        relevant += len(example.term_texts)
    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant
    f1 = (2 * p * r) / (p + r) if p > 0 and r > 0 else 0.
    return {'p': p, 'r': r, 'f1': f1, 'common': common, 'retrieved': retrieved, 'relevant': relevant}, all_nbest_json

def eval_mate(all_examples, all_features, all_results, do_lower_case, verbose_logging, logger):
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_nbest_json = collections.OrderedDict()
    common, relevant, retrieved = 0., 0., 0.
    
    for (feature_index, feature) in enumerate(all_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        pred_terms = []

        for start_index, end_index,  span_mask in \
                zip(result.start_indexes, result.end_indexes,  result.span_masks):
            if span_mask:
                final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                    do_lower_case, verbose_logging, logger)
                pred_terms.append(final_text)


        prediction = {'pred_terms': pred_terms,'gold_terms':example.term_texts}
        all_nbest_json[example.example_id] = prediction

        for term in pred_terms:
            common+= metric_mate(exact_match_score, term, example.term_texts)
            
        retrieved += len(pred_terms)
        relevant += len(example.term_texts)
    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant
    f1 = (2 * p * r) / (p + r) if p > 0 and r > 0 else 0.
    return {'mate_p': p, 'mate_r': r, 'mate_f1': f1, 'mate_common': common, 'mate_retrieved': retrieved, 'mate_relevant': relevant}, all_nbest_json

def eval_masc(all_examples, all_features, all_results, do_lower_case, verbose_logging, logger):
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_nbest_json = collections.OrderedDict()
    common, relevant, retrieved = 0., 0., 0.
    matched_predictions = []
    matched_ground_truth = []

    for (feature_index, feature) in enumerate(all_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        input_terms = []
        pred_polarities = []
        
        for start_index, end_index, cls_pred, span_mask in \
                zip(result.start_indexes, result.end_indexes, result.cls_pred, result.span_masks):

            if span_mask:
                final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                    do_lower_case, verbose_logging, logger)
                input_terms.append(final_text)
                pred_polarities.append(id_to_label[cls_pred])

        prediction = {'input_terms': input_terms, 'pred_polarities': pred_polarities,'gold_terms':example.term_texts,'gold_polarites':example.polarities}
        all_nbest_json[example.example_id] = prediction

        for term, polarity in zip(input_terms, pred_polarities):
            try:
                index = example.term_texts.index(term)
                matched_predictions.append(polarity)
                matched_ground_truth.append(example.polarities[index])
            except ValueError:
                # Handle the case where the term is not found in ground truth
                pass
    
    retrieved = len(matched_predictions)
    relevant = len(matched_ground_truth)
    acc = accuracy_score(matched_ground_truth, matched_predictions)
    macro_f1 = f1_score(matched_ground_truth, matched_predictions, average='macro') 

    return {'masc_acc': acc, 'masc_f1': macro_f1, 'masc_retrieved': retrieved, 'masc_relevant': relevant}, all_nbest_json

def eval_masc_micro(all_examples, all_features, all_results, do_lower_case, verbose_logging, logger):
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_nbest_json = collections.OrderedDict()
    common, relevant, retrieved = 0., 0., 0.
    
    for (feature_index, feature) in enumerate(all_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        input_terms = []
        pred_polarities = []
        
        for start_index, end_index, cls_pred, span_mask in \
                zip(result.start_indexes, result.end_indexes, result.cls_pred, result.span_masks):

            if span_mask:
                final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                    do_lower_case, verbose_logging, logger)
                input_terms.append(final_text)
                pred_polarities.append(id_to_label[cls_pred])

        prediction = {'input_terms': input_terms, 'pred_polarities': pred_polarities,'gold_terms':example.term_texts,'gold_polarites':example.polarities}
        all_nbest_json[example.example_id] = prediction

        for term, polarity in zip(input_terms, pred_polarities):
            # print("term:",term)
            # print('all terms:',example.term_texts)
            # print("polarity:",polarity)
            # print("all polarities:",example.polarities)
            common += masc_metric(exact_match_score, term, polarity, example.term_texts, example.polarities)
            
        retrieved += len(pred_polarities)
        relevant += len(example.polarities)

    acc = common / relevant if relevant > 0 else 0.

    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant if relevant > 0 else 0.
    f1 = (2 * p * r) / (p + r) if p > 0 and r > 0 else 0.
    return {'masc_acc': acc, 'masc_f1': f1, 'masc_common': common, 'masc_retrieved': retrieved, 'masc_relevant': relevant}, all_nbest_json


def evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=False):
    all_results = []
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, span_starts, span_ends, label_masks, example_indices = batch
        with torch.no_grad():
            cls_logits = model('inference', input_mask, input_ids=input_ids, token_type_ids=segment_ids,
                               span_starts=span_starts, span_ends=span_ends)

        for j, example_index in enumerate(example_indices):
            cls_pred = cls_logits[j].detach().cpu().numpy().argmax(axis=1).tolist()
            start_indexes = span_starts[j].detach().cpu().tolist()
            end_indexes = span_ends[j].detach().cpu().tolist()
            span_masks = label_masks[j].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawFinalResult(unique_id=unique_id, start_indexes=start_indexes,
                                              end_indexes=end_indexes, cls_pred=cls_pred, span_masks=span_masks))

    metrics, all_nbest_json = eval_absa(eval_examples, eval_features, all_results,
                                        args.do_lower_case, args.verbose_logging, logger)

    if write_pred:
        output_file = os.path.join(args.output_dir, "predictions.json")
        with open(output_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        logger.info("Writing predictions to: %s" % (output_file))
    return metrics

if __name__=='__main__':
    eval_absa
