import os
from collections import Counter
from itertools import chain
from logging import getLogger

import hydra
import seqeval.metrics

root_path = os.getcwd()
logger = getLogger(__name__)


def majority(preds):
    count = Counter(preds)
    count = count.most_common()

    return count[0][0]


def upper_bound(label, preds):
    if label in preds:
        return label
    else:
        return majority(preds)


def get_const(preds):
    return preds[-1]


def round_table(file_iter, vote="majority"):
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
                eval_labels.append(eval_label)
                eval_pred = []
                eval_label = []
            continue

        line = line.split(" ")

        label = line[1]
        preds = line[2:]

        if vote == "majority":
            pred = majority(preds)
        elif vote == "upper_bound":
            pred = upper_bound(label, preds)
        elif vote == "const":
            pred = get_const(preds)
        else:
            print("Vote Error")
            exit(1)
        eval_pred.append(pred)
        eval_label.append(label)

    eval_preds = list(chain.from_iterable(eval_preds))
    eval_labels = list(chain.from_iterable(eval_labels))
    logger.info("\n" + seqeval.metrics.classification_report([eval_labels], [eval_preds], digits=4))


@hydra.main(config_path="../config", config_name="conll2003", version_base="1.1")
def main(cfg):
    if cfg.test == 2003:
        encoding = "cp-932"
    else:
        encoding = "utf-8"

    input_path = os.path.join(root_path, "outputs100\\Bert\\Reg3test\\many_preds.txt")
    with open(input_path, encoding=encoding) as f:
        text = f.read()
    with open("./input_pred.txt", "w", encoding=encoding) as f:
        f.write(text)
    file_iter = text.split("\n")
    round_table(file_iter, "upper_bound")


if __name__ == "__main__":
    main()
