This repository implements and evaluates different models for sentiment analysis on the IMDB movie review dataset, comparing traditional machine learning methods with modern deep learning. 

It was the submission for 7.2—Transformers for the NLP HD task for SIT330.

# Section 1: Setup

### 1.1 Hardware

All experiments were done using 1xNVIDIA RTX 4090.

### 1.2 Dependencies

```bash
git clone https://github.com/brandonsmith301/transformers-for-nlp.git
cd transformers-for-nlp
pip install -r requirements.txt
```

### 1.3 Dataset

This project uses the IMDB dataset, which is automatically downloaded via the Hugging Face datasets library during the first run of any training script.

# Section 2: Training

Training scripts and configurations are located in the code and conf folders respectively. We leverage the Hugging Face accelerate library for distributed training.

## 2.1 Traditional Models

```bash
cd code
python train_traditional.py
```

This trains all four traditional models specified in the configuration `conf/traditional/config_traditional.yaml`.
Code implementation can be found in `code/traditional`.

## 2.2 LSTM Model

```bash
# For Single-GPU
cd code
python train_lstm.py

# For multi-GPU 
# accelerate config
# accelerate launch train_lstm.py
```
Key configuration parameters in `conf/lstm/config_lstm.yaml` and code implementation can be found in `code/lstm`.

## 2.3 BERT Model

```bash
# For Single-GPU
cd code
python train_bert.py

# For multi-GPU 
# accelerate config
# accelerate launch train_bert.py
```

Key configuration parameters in `conf/lstm/config_bert.yaml` and code implementation can be found in `code/bert`.

## 2.4 Custom Models

```bash
cd code
# Mean Pooling
python train_custom.py --config-name=config_custom_mean_pooling

# Max Pooling
python train_custom.py --config-name=config_custom_max_pooling

# Min Pooling
python train_custom.py --config-name=config_custom_min_pooling

# Mean-Max Pooling
python train_custom.py --config-name=config_custom_mean_max_pooling

# LSTM Pooling
python train_custom.py --config-name=config_custom_lstm_pooling

# GRU Pooling
python train_custom.py --config-name=config_custom_gru_pooling

# Attention Pooling
python train_custom.py --config-name=config_custom_attn_pooling

# Weighted Layer Pooling
python train_custom.py --config-name=config_custom_weighted_pooling

# Concat Pooling
python train_custom.py --config-name=config_custom_concat_pooling

# GeM Pooling
python train_custom.py --config-name=config_custom_gem_pooling
```
