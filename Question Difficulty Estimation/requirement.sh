#!/bin/bash
wget 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'

unzip uncased_L-12_H-768_A-12.zip
rm -rf *.zip

pip install torch
pip install torchvision
pip install tensorflow
pip install tqdm
pip install numpy

export BERT_BASE_DIR='uncased_L-12_H-768_A-12'
python convert_tf_checkpoint_to_pytorch.py \
	--tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
	--bert_config_file $BERT_BASE_DIR/bert_config.json \
	--pytorch_dump_path $BERT_BASE_DIR/pytorch_model.bin

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
