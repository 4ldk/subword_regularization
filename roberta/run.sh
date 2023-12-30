#!/bin/sh

set -eux

cd roberta
pip install --user -r requirements.txt
export PATH=~/.local/bin:$PATH

python3 ./run.py