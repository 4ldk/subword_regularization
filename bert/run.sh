#!/bin/sh

set -eux

cd subword_regularization
pip install --user -r requirements.txt
export PATH=~/.local/bin:$PATH

model_name="bert-base-cased"

sed -z "s/\nmodel_name: /\nmodel_name:$model_name  #/" ./config/conll2003.yaml > ./config/conll2003_modeled.yaml
python3 ./bert/run.py --config conll2003_modeled

sed -z "s/\np: 0/\np: 0.1/" ./config/conll2003_modeled.yaml > ./config/conll2003_01.yaml
python3 ./bert/run.py --config conll2003_01

sed -z "s/\np: 0/\np: 0.3/" ./config/conll2003_modeled.yaml > ./config/conll2003_03.yaml
python3 ./bert/run.py --config conll2003_03
  