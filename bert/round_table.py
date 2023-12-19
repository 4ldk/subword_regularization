from collections import Counter

import conlleval
import hydra


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
    output = []
    for line in file_iter:
        if line[-1:] == "\n":
            line = line[:-1]
        if len(line) < 5:
            output.append("")
        else:
            line = line.split(" ")
            token = line[0]
            pos = line[1]
            label = line[2]
            preds = line[3:]

            if vote == "majority":
                pred = majority(preds)
            elif vote == "upper_bound":
                pred = upper_bound(label, preds)
            else:
                print("Vote Error")
                exit(1)
            output.append(" ".join([token, pos, label, pred]))

    output = "\n".join(output)
    with open("./output.txt", "w", encoding=encoding) as f:
        f.write(output)

    with open("./output.txt", encoding=encoding) as f:
        file_iter = f.readlines()

    conlleval.evaluate_conll_file(file_iter)


@hydra.main(config_path="../config", config_name="conll2003", version_base="1.1")
def main(cfg):
    if cfg.test == 2003:
        encoding = "cp-932"
    else:
        encoding = "utf-8"

    input_path = "C:/Users/chenr/Desktop/python/subword_regularization/outputs/bert_pred_roop/2023-12-19/bertner_large_03_test/many_preds.txt"
    with open(input_path, encoding=encoding) as f:
        text = f.read()
    with open("./input_pred.txt", "w", encoding=encoding) as f:
        f.write(text)
    file_iter = text.split("\n")
    round_table(file_iter, encoding, "majority")


if __name__ == "__main__":
    main()
