# JCC
# ICME2024_MML4SG
# Joint Modal Circular Complementary Attention for Multimodal Aspect-Based Sentiment Analysis

This is the official PyTorch implementation for the paper: **"Joint Modal Circular Complementary Attention for Multimodal Aspect-Based Sentiment Analysis"**.
### [Paper](https://ieeexplore.ieee.org/document/10645483)

## 1. Overview
Existing approaches to Multimodal Aspect-Based Sentiment Analysis have drawbacks: (i) Aspect extraction and sentiment classification always exhibit loose connections, overlooking aspect correlations which leads to inaccurate analysis of indirectly described aspects. (ii) Image pixels are coarsely treated equally in most methods, introducing visual noise that compromise sentiment analysis accuracy. (iii) Additionally, most
rely on extra pre-training image-text relation detection networks, limiting their generality. To address these issues, we propose the Joint modal Circular Complementary attention framework (JCC) which optimizes aspect extraction and sentiment classification jointly by incorporating global text to enhance the model’s awareness of aspect correlations. JCC utilizes text for visual highlighting to mitigate the impact of visual noise. Furthermore, we design the Circular Attention module (CIRA) for general featurefocused aspect extraction and the Modal Complementary Attention module (MCA) for detailed information-focused sentiment classification. Experimental results across three MABSA subtasks demonstrate the superiority of JCC over existing methods.

## 2. Environment Setup
```bash
# Create the conda environment (we use Python 3.9)
conda create -n JCC python=3.9 -y
# Activate the environment
conda activate JCC
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

## 3. Dataset and Checkpoints

Our dataset has been uploaded to Google Drive: [https://drive.google.com/drive/folders/1TJVFZltNUOVyIYzhujSO3ajixtkwVT8i?usp=drive_link](https://drive.google.com/drive/folders/1TJVFZltNUOVyIYzhujSO3ajixtkwVT8i?usp=drive_link)

The dataset structure is as follows:

* `Twitter/twitter15_pre`: The text portion of the Twitter 2015 dataset.
* `Twitter/twitter17_pre`: The text portion of the Twitter 2017 dataset.
* `Twitter/twitter2015_images`: The image portion of the Twitter 2015 dataset.
* `Twitter/twitter2017_images`: The image portion of the Twitter 2017 dataset.

After downloading `resnet152.pth`, please place it under the `resnet/` directory.

## 4. Usage
### Training

First, navigate to the `absa` directory. To start training, run the following command:

```bash
python main.py \
    --output_dir ../tt17/1201/ \
    --train_file train.txt \
    --predict_file dev.txt \
    --data_dir ../data/Twitter/twitter17_pre \
    --image_path ../data/Twitter/twitter2017_images/ \
    --do_train \
    --seed 42 \
    --gpu_idx 7 \
    --num_train_epochs 50.0 \
    --gradient_accumulation_steps 16 \
    --train_batch_size 128
```

### Testing
To evaluate the model, run the following command:

```bash
python main.py \
    --output_dir ../tt15/1201/ \
    --predict_file test.txt \
    --data_dir ../data/Twitter/twitter15_pre \
    --image_path ../data/Twitter/twitter2015_images/ \
    --do_predict \
    --seed 42 \
    --gpu_idx 6
```

## 5. Citation
If you find our work helpful for your research, please consider citing our paper:

```bibtex
@inproceedings{liu2024joint,
  title={Joint Modal Circular Complementary Attention for Multimodal Aspect-Based Sentiment Analysis},
  author={Liu, Hao and He, Lijun and Liang, Jiaxi},
  booktitle={2024 IEEE International Conference on Multimedia and Expo Workshops (ICMEW)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```

Finally, if you encounter any issues or have questions, feel free to open an issue.
