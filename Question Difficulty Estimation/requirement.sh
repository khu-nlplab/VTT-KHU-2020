#!/bin/bash
pip install torch
pip install torchvision
pip install tensorflow
pip install tqdm
pip install numpy

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
