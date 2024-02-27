#!/bin/bash

conda install --yes pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torchmetrics
conda install --yes matplotlib
pip install accelerate
pip install transformers
pip install timm
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install tensorboard
pip install python-box
conda install --yes pandas
pip install peft
pip install -U scikit-learn
pip install fair-esm
