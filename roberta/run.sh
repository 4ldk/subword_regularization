#!/bin/sh

set -eux
conda create -n py310 python=3.10
conda init bash
conda activate py310

cd roberta
apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
pip install --user -r requirements.txt
export PATH=~/.local/bin:$PATH

python ./run.py