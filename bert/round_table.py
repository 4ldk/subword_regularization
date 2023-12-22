from collections import Counter
import sys
import os
import seqeval.metrics
from itertools import chain

import hydra
from sklearn.metrics import classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.boi_convert import boi1_to_2


def majority(preds):
    count = Counter(preds)
    count = count.most_common()

    return count[0][0]


def upper_bound(label, preds):
    if label in preds:
        return label
    else:
        return majority(preds)


def round_table(file_iter, encoding, vote="majority"):
    eval_preds = []
    eval_labels = []

    eval_pred = []
    eval_label = []
    for line in file_iter:
        if line[-1:] == "\n":
            line = line[:-1]

        if len(line) < 5:
            if len(eval_pred) != 0:
                eval_preds.append(eval_pred)
                eval_label = boi1_to_2(eval_label)
                eval_labels.append(eval_label)
                eval_pred = []
                eval_label = []
            continue

        line = line.split(" ")

        label = line[2]
        preds = line[3:]

        if vote == "majority":
            pred = majority(preds)
        elif vote == "upper_bound":
            pred = upper_bound(label, preds)
        else:
            print("Vote Error")
            exit(1)
        eval_pred.append(pred)
        eval_label.append(label)

    eval_preds = list(chain.from_iterable(eval_preds))
    eval_labels = list(chain.from_iterable(eval_labels))
    print(seqeval.metrics.classification_report([eval_labels], [eval_preds], digits=4))
    print(classification_report(eval_labels, eval_preds, digits=4))


@hydra.main(config_path="../config", config_name="conll2003", version_base="1.1")
def main(cfg):
    if cfg.test == 2003:
        encoding = "cp-932"
    else:
        encoding = "utf-8"

    input_path = (
        "C:/Users/chenr/Desktop/python/subword_regularization/outputs/bert_pred_roop/2023-12-22/19-05-42/many_preds.txt"
    )
    with open(input_path, encoding=encoding) as f:
        text = f.read()
    with open("./input_pred.txt", "w", encoding=encoding) as f:
        f.write(text)
    file_iter = text.split("\n")
    round_table(file_iter, encoding, "majority")


if __name__ == "__main__":
    main()
