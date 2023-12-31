#!/bin/sh

set -eux

cd subword_regularization
pip install --user -r requirements.txt
export PATH=~/.local/bin:$PATH

model_name="roberta-large"

sed -z "s/\nmodel_name: /\nmodel_name:$model_name  #/" ./config/conll2003.yaml > ./config/conll2003_roberta.yaml
sed -i -z "/\nlength: /\nlength: 510 #/" ./config/conll2003_roberta.yaml
python3 ./roberta/run.py --config conll2003_roberta

sed -z "s/\np: 0/\np: 0.1/" ./config/conll2003_roberta.yaml > ./config/conll2003_roberta01.yaml
python3 ./roberta/run.py --config conll2003_roberta01

sed -z "s/\np: 0/\np: 0.3/" ./config/conll2003_roberta.yaml > ./config/conll2003_roberta03.yaml
python3 ./roberta/run.py --config conll2003_roberta03