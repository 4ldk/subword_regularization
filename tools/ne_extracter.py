import sys
import os

from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.maxMatchTokenizer import MaxMatchTokenizer
from utils.utils import path_to_data, val_to_key


def pred(data):
    if data == "train":
        dataset = load_dataset("conll2003")
        test_dataset = dataset["train"]
        encoding = "cp932"

    if data == "2003":
        dataset = load_dataset("conll2003")
        test_dataset = dataset["test"]
        encoding = "cp932"

    elif data == "2023":
        test_dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/conll2023.txt")
        encoding = "utf-8"

    elif data == "crossweigh":
        test_dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/conllcw.txt")
        encoding = "utf-8"

    ner_dict = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-MISC": 7,
        "I-MISC": 8,
        "PAD": 9,
    }

    mmt = MaxMatchTokenizer(ner_dict=ner_dict, p=0)
    bert_tokeninzer = AutoTokenizer.from_pretrained("bert-base-cased")
    mmt.loadBertTokenizer(bertTokenizer=bert_tokeninzer)

    out_tokens, out_ners = (
        test_dataset["tokens"],
        test_dataset["ner_tags"],
    )

    out = {}
    for out_token, out_ner in zip(out_tokens, out_ners):

        for tk, ner in zip(out_token, out_ner):

            tag = val_to_key(ner, ner_dict)
            if tag == "O":
                continue
            if tk not in out.keys():
                upper = tk.isupper()
                subword_len = len(mmt.tokenizeWord(tk))
                subword = " ".join(mmt.tokenizeWord(tk))

                if data != "train":
                    with open("./ne_data_train.txt") as f:
                        train_data = f.read().split("\n")
                        train_data = [train_row.split(", ")[0] for train_row in train_data]

                    in_train = tk in train_data
                    out[tk] = [1, upper, in_train, subword, subword_len]
                else:
                    out[tk] = [1, upper, subword, subword_len]
            else:
                out[tk][0] += 1

    out = sorted(out.items(), key=lambda x: x[1][0], reverse=True)
    text = ""
    for k, v in out:
        text += k + ", "
        for vv in v:
            text += str(vv)
            text += ", "
        text += "\n"

    with open("./ne_data.txt", "w", encoding=encoding) as f:
        f.write(text)


if __name__ == "__main__":
    pred("2003")
