import copy
import os


def convert(texts, output_path, encoding="utf-8"):
    new_dataset = []
    pre_tag = ""
    for d in texts:
        if "-DOCSTART-" in d:
            continue
        if len(d) == 0:
            new_dataset.append(d)
            pre_tag = ""

        else:
            data = d.split()
            if data[-1][0] == "I" and pre_tag in ["", "O"]:
                data[-1] = list(data[-1])
                data[-1][0] = "B"
                data[-1] = "".join(data[-1])
                d = " ".join(data)

            pre_tag = data[-1][0]
            new_dataset.append(d)

    new_dataset = "\n".join(new_dataset)
    with open(output_path, "w", encoding=encoding) as f_out:
        f_out.write(new_dataset)


def boi2_to_1(dataset):
    tokens = dataset["tokens"]
    pos_tags = dataset["pos_tags"]
    ner_tags = dataset["ner_tags"]

    new_tags = []
    for ner in ner_tags:
        # ner = [O,B,I,I,I,I]
        pre_tag = -1
        new_ner = []
        for n in ner:
            if pre_tag in [-1, 0] and n in [1, 3, 5, 7]:
                pre_tag = copy.deepcopy(n)
                n += 1
            else:
                pre_tag = copy.deepcopy(n)
            new_ner.append(copy.deepcopy(n))
        # new_ner = [O,I,I,I,I,I]
        new_tags.append(new_ner)

    new_dataset = {
        "tokens": tokens,
        "ner_tags": new_tags,
        "pos_tags": pos_tags,
    }
    return new_dataset


if __name__ == "__main__":
    if not os.path.exists("./test_datasets"):
        os.makedirs("./test_datasets")

    encoding = "utf-8"
    with open("./CrossWeigh/data/conllpp_test.txt", encoding=encoding) as f:
        text = f.read()
        text = text.split("\n")
        text = text[2:]
    convert(text, "./test_datasets/conllcw.txt", encoding)

    with open("./acl2023_conllpp/dataset/conllpp.txt", encoding=encoding) as f:
        text = f.read()
        text = text.split("\n")
        text = text[2:]
    convert(text, "./test_datasets/conll2023.txt", encoding)
