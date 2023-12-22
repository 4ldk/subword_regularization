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
            if data[-1][0] == "I":
                if pre_tag in ["", "O"]:
                    data[-1] = "B" + data[-1][1:]
                elif (len(pre_tag) > 4) and (pre_tag[-3:] != data[-1][-3:]):
                    data[-1] = "B" + data[-1][1:]

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
        # ner = [O,B-LOC,I-LOC,I-LOC,B-LOC,I-LOC,B-MISC,B-MISC]
        pre_tag = -1
        new_ner = []
        for n in ner:
            if n in [1, 3, 5, 7] and pre_tag not in [n + 1, n]:
                pre_tag = copy.deepcopy(n)
                n += 1
            else:
                pre_tag = copy.deepcopy(n)
            new_ner.append(copy.deepcopy(n))
        # new_ner = [O,I-LOC,I-LOC,I-LOC,B-LOC,I-LOC,I-MISC,B-MISC]
        new_tags.append(new_ner)

    new_dataset = {
        "tokens": tokens,
        "ner_tags": new_tags,
        "pos_tags": pos_tags,
    }
    return new_dataset


def boi1_to_2(labels):
    new_labels = []
    pre_tag = ""
    for label in labels:
        if label[0] == "I":
            if pre_tag in ["", "O"]:
                label = "B" + label[1:]
            elif len(pre_tag) > 4 and pre_tag[-3:] != label[-3:]:
                label = "B" + label[1:]

        new_labels.append(label)
        pre_tag = label

    return new_labels


if __name__ == "__main__":
    labels = ["I-ORG", "I-ORG", "O", "I-ORG", "I-MISC", "I-LOC"]
    print(boi1_to_2(labels))
