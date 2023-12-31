#!/bin/sh

set -eux

cd roberta
pip install --user -r requirements.txt
export PATH=~/.local/bin:$PATH

python3 ./run.py

sed -z "s/\np: 0/\np: 0.1/" ./config/conll2003.yaml > ./config/conll2003_01.yaml
python3 ./run.py --config conll2003_01

sed -z "s/\np: 0/\np: 0.3/" ./config/conll2003.yaml > ./config/conll2003_03.yaml
python3 ./run.py --config conll2003_03
  