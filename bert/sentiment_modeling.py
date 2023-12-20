from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from bert.modeling import BertModel, BERTLayerNorm
import torchvision
import resnet.resnet as resnet
from resnet.resnet_utils import myResnet
import numpy as np
import os
from torch.nn import functional as F
from pdb import set_trace as stop
from transformers import RobertaModel

def flatten(x):
    if len(x.size()) == 2:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        return x.view([batch_size * seq_length])
    elif len(x.size()) == 3:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        hidden_size = x.size()[2]
        return x.view([batch_size * seq_length, hidden_size])
    else:
        raise Exception()

def reconstruct(x, ref):
    if len(x.size()) == 1:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        return x.view([batch_size, turn_num])
    elif len(x.size()) == 2:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        sequence_length = x.size()[1]
        return x.view([batch_size, turn_num, sequence_length])
    else:
        raise Exception()

def flatten_emb_by_sentence(emb, emb_mask):
    batch_size = emb.size()[0]
    seq_length = emb.size()[1]
    flat_emb = flatten(emb)
    flat_emb_mask = emb_mask.view([batch_size * seq_length])
    return flat_emb[flat_emb_mask.nonzero().squeeze(), :]

def get_span_representation(span_starts, span_ends, input, input_mask):
    '''
    :param span_starts: [N, M]
    :param span_ends: [N, M]
    :param input: [N, L, D]
    :param input_mask: [N, L]
    :return: [N*M, JR, D], [N*M, JR]
    '''
    input_mask = input_mask.to(dtype=span_starts.dtype)  # fp16 compatibility
    input_len = torch.sum(input_mask, dim=-1) # [N]
    word_offset = torch.cumsum(input_len, dim=0) # [N]
    word_offset -= input_len

    span_starts_offset = span_starts + word_offset.unsqueeze(1)
    span_ends_offset = span_ends + word_offset.unsqueeze(1)

    span_starts_offset = span_starts_offset.view([-1])  # [N*M]
    span_ends_offset = span_ends_offset.view([-1])

    span_width = span_ends_offset - span_starts_offset + 1
    JR = torch.max(span_width)

    context_outputs = flatten_emb_by_sentence(input, input_mask)  # [<N*L, D]
    text_length = context_outputs.size()[0]

    span_indices = torch.arange(JR).unsqueeze(0).to(span_starts_offset.device) + span_starts_offset.unsqueeze(1)  # [N*M, JR]
    span_indices = torch.min(span_indices, (text_length - 1)*torch.ones_like(span_indices))
    span_text_emb = context_outputs[span_indices, :]    # [N*M, JR, D]

    row_vector = torch.arange(JR).to(span_width.device)
    span_mask = row_vector < span_width.unsqueeze(-1)   # [N*M, JR]
    return span_text_emb, span_mask

def get_self_att_representation(input, input_score, input_mask):
    '''
    :param input: [N, L, D]
    :param input_score: [N, L]
    :param input_mask: [N, L]
    :return: [N, D]
    '''
    input_mask = input_mask.to(dtype=input_score.dtype)  # fp16 compatibility
    input_mask = (1.0 - input_mask) * -10000.0
    input_score = input_score + input_mask
    input_prob = nn.Softmax(dim=-1)(input_score)
    input_prob = input_prob.unsqueeze(-1)
    output = torch.sum(input_prob * input, dim=1)
    return output

def distant_cross_entropy(logits, positions, mask=None):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    :param mask: [N]
    '''
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    if mask is not None:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(dtype=log_probs.dtype)))
    else:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss

def pad_sequence(sequence, length):
    while len(sequence) < length:
        sequence.append(0)
    return sequence

def convert_crf_output(outputs, sequence_length, device):
    predictions = []
    for output in outputs:
        pred = pad_sequence(output[0], sequence_length)
        predictions.append(torch.tensor(pred, dtype=torch.long))
    predictions = torch.stack(predictions, dim=0)
    if device is not None:
        predictions = predictions.to(device)
    return predictions


class JointMMwithRel(nn.Module):
    def __init__(self, mode, config):
        super(JointMMwithRel, self).__init__()
        if mode == 'bert':
            self.bert = BertModel(config)
        elif mode == 'roberta':
            self.bert = RobertaModel(config,add_pooling_layer=False)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.unary_affine = nn.Linear(config.hidden_size, 1)
        self.binary_affine = nn.Linear(config.hidden_size, 2)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 4)

        # resnet
        # self.align_res=nn.Linear(2048,config.hidden_size)
        # self.res2span=MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)

        # # image labels
        
        # self.img_label_emb=nn.Embedding(100,config.hidden_size)
        # self.imgLa2text=MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)
        # self.align_imgLA_and_text=nn.Linear(config.hidden_size*2,config.hidden_size)
        # self.dropout_label=nn.Dropout(0.1)

        # # text2span
        # self.text2span=MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)

        # text image attention wzq
        # self.tiattention1=MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)
        # self.tiattention2=MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)

        # all
        # self.align_all_1=nn.Linear(2*config.hidden_size,2*config.hidden_size)
        # self.align_all_2=nn.Linear(2*config.hidden_size,1*config.hidden_size)
        # self.dropout_all=nn.Dropout(0.1)

        # # resnet
        self.align_res=nn.Linear(2048,config.hidden_size)
        
        # helight
        self.helight = MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)
        
        # cycle trans
        self.cycle = CycleAttention(8, config.hidden_size)
        
        self.MLP1=nn.Linear(config.hidden_size*2,config.hidden_size)
        self.dropout_label=nn.Dropout(0.1)

        # polarity classification
        self.img2aspect = MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)
        self.aspect2txt = MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)
        self.aspect2img = MultiHeadAttention(8,config.hidden_size,config.hidden_size,config.hidden_size)

        self.MLP2 = nn.Linear(2*config.hidden_size,2*config.hidden_size)
        self.MLP3 = nn.Linear(2*config.hidden_size,config.hidden_size)
        self.dropout_all=nn.Dropout(0.1)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)

        # image infomation
        net = getattr(resnet, 'resnet152')()
        net.load_state_dict(torch.load('../resnet/resnet152.pth'))
        self.resnet152 = myResnet(net,True)#True

        self.apply(init_weights)
        
    def forward(self, mode, attention_mask, input_ids=None, token_type_ids=None, 
                start_positions=None, end_positions=None, span_starts=None, 
                span_ends=None, polarity_labels=None, label_masks=None, 
                sequence_input=None,image_labels=None,image_raw_data=None,
                ):

        _, _,enc_img_input_att=self.resnet152(image_raw_data)
        enc_img=enc_img_input_att.view(-1, 2048, 49).permute(0, 2, 1)
        res_feture=self.align_res(enc_img)#[N,L,D]

        if mode == 'train':
            batch_size=input_ids.size()[0]
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output_raw = all_encoder_layers[-1] #[N,L,D]

            # JML
            # imgLa_and_text,_=self.imgLa2text(sequence_output_raw,res_feture,res_feture)
            # sequence_output =self.dropout_label(self.align_imgLA_and_text(torch.cat([sequence_output_raw,imgLa_and_text],2)))
            
            ###################################################################
            # wzq method
            # res_feture_R,_=self.tiattention1(res_feture,sequence_output_raw,sequence_output_raw)  # img&text CA      
            # imgLa_and_text,_=self.imgLa2text(sequence_output_raw,res_feture_R,res_feture_R)  # imgLa_and_text：fusion feature
            # sequence_output =self.dropout_label(self.align_imgLA_and_text(torch.cat([sequence_output_raw,imgLa_and_text],2))) # fusion/img_text concat
            ###################################################################

            ###################################################################
            # new
            # t2t,_ = self.txt2txt(sequence_output_raw,sequence_output_raw,sequence_output_raw)
            # i2i,_ = self.img2img(res_feture,res_feture,res_feture)
            # i2t,_ = self.img2txt(i2i,t2t,t2t)
            # t2i,_ = self.txt2img(t2t,i2i,i2i)

            # sequence_output = self.dropout_label(self.MLP1(torch.cat([self.attention1(t2i,i2t,i2t)[0],sequence_output_raw],2)))
            ###################################################################

            ###################################################################
            # cycle attn
            helight_img,_ = self.helight(sequence_output_raw, res_feture, res_feture)
            output = self.cycle(sequence_output_raw, res_feture, helight_img)
            sequence_output = self.dropout_label(self.MLP1(output))
            ###################################################################

            ae_logits = self.binary_affine(sequence_output)   # [N, L, 2]
            start_logits, end_logits = ae_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            start_loss = distant_cross_entropy(start_logits, start_positions)
            end_loss = distant_cross_entropy(end_logits, end_positions)
            ae_loss = (start_loss + end_loss) / 2

            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]
            span_score = self.unary_affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)

            # JML
            # span_image_output = flatten(self.res2span(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),res_feture,res_feture)[0])
            # span_pooled_output = self.align_all_2(self.dropout_all(self.align_all_1(torch.cat([span_pooled_output,span_image_output],-1))))

            ###################################################################
            # wzq method
            ### take spanfeature as K,V
            # res_feature_V,_ = self.tiattention2(res_feture,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            # span_image_output = flatten(self.res2span(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),res_feature_V,res_feature_V)[0])
            # span_text_output = flatten(self.text2span(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),sequence_output_raw,sequence_output_raw)[0])
            # '''
            # spanimgout: torch.Size([80, 768])
            # spantextout: torch.Size([80, 768])
            # '''
            # span_pooled_output = self.align_all_2(self.dropout_all(self.align_all_1(torch.cat([span_image_output,span_text_output],-1))))
            ###################################################################

            ###################################################################
            # new
            # t2a,_ = self.txt2aspect(sequence_output_raw,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            # i2a,_ = self.img2aspect(res_feture,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            # a2t,_ = self.aspect2txt(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),t2a,t2a)
            # a2i,_ = self.aspect2img(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),i2a,i2a)

            # aspect2text = flatten(a2t)
            # aspect2img = flatten(a2i)
            # span_pooled_output = self.MLP3(self.dropout_all(self.MLP2(torch.cat([aspect2text,aspect2img],-1))))
            ###################################################################

            ###################################################################
            # cycle masc
            i2a,_ = self.img2aspect(helight_img,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            a2t,_ = self.aspect2txt(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),sequence_output_raw,sequence_output_raw)
            a2i,_ = self.aspect2img(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),i2a,i2a)
            aspect2text = flatten(a2t)
            aspect2img = flatten(a2i)
            span_pooled_output = self.MLP3(self.dropout_all(self.MLP2(torch.cat([aspect2text,aspect2img],-1))))
            ###################################################################

            ac_logits = self.classifier(span_pooled_output)  # [N*M, 5]

            ac_loss_fct = CrossEntropyLoss(reduction='none')
            flat_polarity_labels = flatten(polarity_labels)
            flat_label_masks = flatten(label_masks).to(dtype=ac_logits.dtype)
            ac_loss = ac_loss_fct(ac_logits, flat_polarity_labels)
            ac_loss = torch.sum(flat_label_masks * ac_loss) / flat_label_masks.sum()
           
            return ae_loss + ac_loss
        
        elif mode == 'train_mate':
            batch_size=input_ids.size()[0]
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output_raw = all_encoder_layers[-1] #[N,L,D]

            # JML
            # imgLa_and_text,_=self.imgLa2text(sequence_output_raw,res_feture,res_feture)
            # sequence_output =self.dropout_label(self.align_imgLA_and_text(torch.cat([sequence_output_raw,imgLa_and_text],2)))
            
            ###################################################################
            # wzq method
            # res_feture_R,_=self.tiattention1(res_feture,sequence_output_raw,sequence_output_raw)  # img&text CA      
            # imgLa_and_text,_=self.imgLa2text(sequence_output_raw,res_feture_R,res_feture_R)  # imgLa_and_text：fusion feature
            # sequence_output =self.dropout_label(self.align_imgLA_and_text(torch.cat([sequence_output_raw,imgLa_and_text],2))) # fusion/img_text concat
            ###################################################################

            ###################################################################
            # new
            # t2t,_ = self.txt2txt(sequence_output_raw,sequence_output_raw,sequence_output_raw)
            # i2i,_ = self.img2img(res_feture,res_feture,res_feture)
            # i2t,_ = self.img2txt(i2i,t2t,t2t)
            # t2i,_ = self.txt2img(t2t,i2i,i2i)

            # sequence_output = self.dropout_label(self.MLP1(torch.cat([self.attention1(t2i,i2t,i2t)[0],sequence_output_raw],2)))
            ###################################################################

            ###################################################################
            # cycle attn
            helight_img,_ = self.helight(sequence_output_raw, res_feture, res_feture)
            output = self.cycle(sequence_output_raw, res_feture, helight_img)
            sequence_output = self.dropout_label(self.MLP1(output))
            ###################################################################

            ae_logits = self.binary_affine(sequence_output)   # [N, L, 2]
            start_logits, end_logits = ae_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            start_loss = distant_cross_entropy(start_logits, start_positions)
            end_loss = distant_cross_entropy(end_logits, end_positions)
            ae_loss = (start_loss + end_loss) / 2
           
            return ae_loss
        
        elif mode == 'train_masc':
            batch_size=input_ids.size()[0]
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output_raw = all_encoder_layers[-1] #[N,L,D]

            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output_raw,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]
            span_score = self.unary_affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)

            # JML
            # span_image_output = flatten(self.res2span(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),res_feture,res_feture)[0])
            # span_pooled_output = self.align_all_2(self.dropout_all(self.align_all_1(torch.cat([span_pooled_output,span_image_output],-1))))

            ###################################################################
            # wzq method
            ### take spanfeature as K,V
            # res_feature_V,_ = self.tiattention2(res_feture,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            # span_image_output = flatten(self.res2span(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),res_feature_V,res_feature_V)[0])
            # span_text_output = flatten(self.text2span(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),sequence_output_raw,sequence_output_raw)[0])
            # span_pooled_output = self.align_all_2(self.dropout_all(self.align_all_1(torch.cat([span_image_output,span_text_output],-1))))
            ###################################################################

            ###################################################################
            # new
            # t2a,_ = self.txt2aspect(sequence_output_raw,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            # i2a,_ = self.img2aspect(res_feture,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            # a2t,_ = self.aspect2txt(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),t2a,t2a)
            # a2i,_ = self.aspect2img(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),i2a,i2a)

            # aspect2text = flatten(a2t)
            # aspect2img = flatten(a2i)
            # span_pooled_output = self.MLP3(self.dropout_all(self.MLP2(torch.cat([aspect2text,aspect2img],-1))))
            ###################################################################

            ###################################################################
            # cycle masc
            helight_img,_ = self.helight(sequence_output_raw, res_feture, res_feture)
            i2a,_ = self.img2aspect(helight_img,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            a2t,_ = self.aspect2txt(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),sequence_output_raw,sequence_output_raw)
            a2i,_ = self.aspect2img(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),i2a,i2a)

            aspect2text = flatten(a2t)
            aspect2img = flatten(a2i)
            span_pooled_output = self.MLP3(self.dropout_all(self.MLP2(torch.cat([aspect2text,aspect2img],-1))))
            ###################################################################

            ac_logits = self.classifier(span_pooled_output)  # [N*M, 5]

            ac_loss_fct = CrossEntropyLoss(reduction='none')
            flat_polarity_labels = flatten(polarity_labels)
            flat_label_masks = flatten(label_masks).to(dtype=ac_logits.dtype)
            ac_loss = ac_loss_fct(ac_logits, flat_polarity_labels)
            ac_loss = torch.sum(flat_label_masks * ac_loss) / flat_label_masks.sum()
           
            return ac_loss

        elif mode == 'extraction':
            
            batch_size=input_ids.size()[0]
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output_raw = all_encoder_layers[-1]

            # JML
            # imgLa_and_text,_=self.imgLa2text(sequence_output_raw,res_feture,res_feture)
            # sequence_output =self.dropout_label(self.align_imgLA_and_text(torch.cat([sequence_output_raw,imgLa_and_text],2)))
            
            ###################################################################
            # wzq
            # res_feture_R,_=self.tiattention1(res_feture,sequence_output_raw,sequence_output_raw)
            # imgLa_and_text,_=self.imgLa2text(sequence_output_raw,res_feture_R,res_feture_R)
            # sequence_output =self.dropout_label(self.align_imgLA_and_text(torch.cat([sequence_output_raw,imgLa_and_text],2)))
            ###################################################################

            ###################################################################
            # # new
            # t2t,_ = self.txt2txt(sequence_output_raw,sequence_output_raw,sequence_output_raw)
            # i2i,_ = self.img2img(res_feture,res_feture,res_feture)
            # i2t,_ = self.img2txt(i2i,t2t,t2t)
            # t2i,_ = self.txt2img(t2t,i2i,i2i)

            # sequence_output = self.dropout_label(self.MLP1(torch.cat([self.attention1(t2i,i2t,i2t)[0],sequence_output_raw],2)))
            ###################################################################

            ###################################################################
            # cycle attn
            helight_img,_ = self.helight(sequence_output_raw, res_feture, res_feture)
            output = self.cycle(sequence_output_raw, res_feture, helight_img)
            sequence_output = self.dropout_label(self.MLP1(output))
            ###################################################################

            ae_logits = self.binary_affine(sequence_output)
            start_logits, end_logits = ae_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            return start_logits, end_logits, sequence_output #,torch.gt(relations,0.5)

        elif mode == 'classification':
            batch_size=sequence_input.size()[0]
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_input,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]

            span_score = self.unary_affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)

            ###################################################################
            # wzq
            # get the info from the text and resnet
            # all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            # sequence_output_raw = all_encoder_layers[-1]
            # res_feature_V,_ = self.tiattention2(res_feture,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            # span_image_output = flatten(self.res2span(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),res_feature_V,res_feature_V)[0])
            # span_text_output = flatten(self.text2span(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),sequence_output_raw,sequence_output_raw)[0])
            # span_pooled_output = self.align_all_2(self.dropout_all(self.align_all_1(torch.cat([span_image_output,span_text_output],-1))))
            ###################################################################

            # get the info from the text and resnet
            # JML
            # span_image_output= flatten(self.res2span(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),res_feture,res_feture)[0])
            # span_pooled_output=self.align_all_2(self.dropout_all(self.align_all_1(torch.cat([span_pooled_output,span_image_output],-1))))
            
            ###################################################################
            # new
            # all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            # sequence_output_raw = all_encoder_layers[-1]

            # t2a,_ = self.txt2aspect(sequence_output_raw,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            # i2a,_ = self.img2aspect(res_feture,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            # a2t,_ = self.aspect2txt(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),t2a,t2a)
            # a2i,_ = self.aspect2img(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),i2a,i2a)
            
            # aspect2text = flatten(a2t)
            # aspect2img = flatten(a2i)
            # span_pooled_output = self.MLP3(self.dropout_all(self.MLP2(torch.cat([aspect2text,aspect2img],-1))))
            ###################################################################

            ###################################################################
            # cycle masc
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output_raw = all_encoder_layers[-1]
            helight_img,_ = self.helight(sequence_output_raw, res_feture, res_feture)
            i2a,_ = self.img2aspect(helight_img,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            a2t,_ = self.aspect2txt(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),sequence_output_raw,sequence_output_raw)
            a2i,_ = self.aspect2img(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),i2a,i2a)

            aspect2text = flatten(a2t)
            aspect2img = flatten(a2i)
            span_pooled_output = self.MLP3(self.dropout_all(self.MLP2(torch.cat([aspect2text,aspect2img],-1))))
            ###################################################################
            ac_logits = self.classifier(span_pooled_output)  # [N*M, 5]

            return reconstruct(ac_logits, span_starts) # , torch.gt(relations,0.5)
        
        elif mode == 'masc':
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output_raw = all_encoder_layers[-1]
            
            batch_size=sequence_output_raw.size()[0]
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output_raw,
                                                             attention_mask)  # [N*M, JR, D], [N*M, JR]

            span_score = self.unary_affine(span_output)
            span_score = span_score.squeeze(-1)  # [N*M, JR]
            span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            
            ###################################################################
            # wzq
            # get the info from the text and resnet
            # res_feature_V,_ = self.tiattention2(res_feture,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            # span_image_output = flatten(self.res2span(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),res_feature_V,res_feature_V)[0])
            # span_text_output = flatten(self.text2span(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),sequence_output_raw,sequence_output_raw)[0])
            # span_pooled_output = self.align_all_2(self.dropout_all(self.align_all_1(torch.cat([span_image_output,span_text_output],-1))))
            ###################################################################

            ###################################################################
            # 1\JML ablation
            # span_image_output= flatten(self.res2span(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),res_feture,res_feture)[0])#[N*M,D]
            # span_pooled_output=self.align_all_2(self.dropout_all(self.align_all_1(torch.cat([span_pooled_output,span_image_output],-1))))
            ###################################################################

            ###################################################################
            # new
            # t2a,_ = self.txt2aspect(sequence_output_raw,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            # i2a,_ = self.img2aspect(res_feture,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            # a2t,_ = self.aspect2txt(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),t2a,t2a)
            # a2i,_ = self.aspect2img(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),i2a,i2a)
            
            # aspect2text = flatten(a2t)
            # aspect2img = flatten(a2i)
            # span_pooled_output = self.MLP3(self.dropout_all(self.MLP2(torch.cat([aspect2text,aspect2img],-1))))
            ###################################################################

            ###################################################################
            # cycle masc
            helight_img,_ = self.helight(sequence_output_raw, res_feture, res_feture)
            i2a,_ = self.img2aspect(helight_img,span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]))
            a2t,_ = self.aspect2txt(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),sequence_output_raw,sequence_output_raw)
            a2i,_ = self.aspect2img(span_pooled_output.reshape(batch_size,-1,span_pooled_output.size()[-1]),i2a,i2a)

            aspect2text = flatten(a2t)
            aspect2img = flatten(a2i)
            span_pooled_output = self.MLP3(self.dropout_all(self.MLP2(torch.cat([aspect2text,aspect2img],-1))))
            ###################################################################
            ac_logits = self.classifier(span_pooled_output)  # [N*M, 5]
            return reconstruct(ac_logits, span_starts)

def distant_loss(start_logits, end_logits, start_positions=None, end_positions=None, mask=None):
    start_loss = distant_cross_entropy(start_logits, start_positions, mask)
    end_loss = distant_cross_entropy(end_logits, end_positions, mask)
    total_loss = (start_loss + end_loss) / 2
    return total_loss

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, dropout2=False,attn_type='softmax'):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k,bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k,bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v,bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        if dropout2:
            # self.dropout2 = nn.Dropout(dropout2)
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,dropout=dropout2)
        else:
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),attn_type=attn_type,dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        

        self.layer_norm = nn.LayerNorm(d_model)

        if n_head > 1:
            self.fc = nn.Linear(n_head * d_v, d_model,bias=False)
            nn.init.xavier_normal_(self.fc.weight)


    def forward(self, q, k, v, attn_mask=None,dec_self=False): 

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        
        if hasattr(self,'dropout2'):
            q = self.dropout2(q)

        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)


        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv


        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, attn_mask=attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        if hasattr(self,'fc'):
            output = self.fc(output)

        if hasattr(self,'dropout'):
            output = self.dropout(output)
        

        if dec_self:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output + residual)

        return output, attn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1, attn_type='softmax'):
        super().__init__()
        self.temperature = temperature  
        self.dropout = nn.Dropout(dropout)
        if attn_type == 'softmax':
            self.attn_type = nn.Softmax(dim=2)
            # self.softmax = BottleSoftmax()
        else:
            self.attn_type = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None,stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            # attn = attn.masked_fill(attn_mask, -np.inf)
            attn = attn.masked_fill(attn_mask, -1e6)

        if stop_sig:
            print('**')
            stop()


        attn = self.attn_type(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

class CycleAttention(nn.Module):
    def __init__(self, n_head, d_model) -> None:
        super().__init__()
        self.n_head = n_head
        self.hidden_dim = d_model

        self.Attn1 = MultiHeadAttention(n_head=self.n_head, d_model=self.hidden_dim, d_k=self.hidden_dim, d_v=self.hidden_dim)
        self.Attn2 = MultiHeadAttention(n_head=self.n_head, d_model=self.hidden_dim, d_k=self.hidden_dim, d_v=self.hidden_dim)
        self.Attn3 = MultiHeadAttention(n_head=self.n_head, d_model=self.hidden_dim, d_k=self.hidden_dim, d_v=self.hidden_dim)
        # self.Attn4 = MultiHeadAttention(n_head=self.n_head, d_model=self.hidden_dim, d_k=self.hidden_dim, d_v=self.hidden_dim)
        # self.Attn5 = MultiHeadAttention(n_head=self.n_head, d_model=self.hidden_dim, d_k=self.hidden_dim, d_v=self.hidden_dim)
        # self.Attn6 = MultiHeadAttention(n_head=self.n_head, d_model=self.hidden_dim, d_k=self.hidden_dim, d_v=self.hidden_dim)
        self.Attn7 = MultiHeadAttention(n_head=self.n_head, d_model=self.hidden_dim, d_k=self.hidden_dim, d_v=self.hidden_dim)

    def forward(self, text_rep, img_rep, helighted_img):
        trans_vt,_ = self.Attn1(helighted_img, text_rep, text_rep)
        trans_hvt,_ = self.Attn2(img_rep, trans_vt, trans_vt)
        trans_thvt,_ = self.Attn3(text_rep, trans_hvt, trans_hvt)

        trans_vt,_ = self.Attn1(img_rep,trans_thvt,trans_thvt)
        trans_hvt,_ = self.Attn2(helighted_img, trans_vt, trans_vt)
        trans_thvt,_ = self.Attn3(text_rep, trans_hvt, trans_hvt)

        # trans_vt,_ = self.Attn4(img_rep,trans_thvt,trans_thvt)
        # trans_hvt,_ = self.Attn5(helighted_img, trans_vt, trans_vt)
        # trans_thvt,_ = self.Attn6(text_rep, trans_hvt, trans_hvt)

        trans_final,_ = self.Attn7(trans_thvt, trans_thvt, trans_thvt)
        trans_final = torch.cat([trans_final, text_rep],2)

        return trans_final