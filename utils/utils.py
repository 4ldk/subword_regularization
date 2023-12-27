import os
import sys

import torch
from seqeval import metrics
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from boi_convert import boi1_to_2

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


class BertDataset(Dataset):
    def __init__(self, X, mask, type_ids, y) -> None:
        super().__init__()
        if type(X) is not torch.Tensor:
            X = torch.tensor(X)
        if type(mask) is not torch.Tensor:
            mask = torch.tensor(mask)
        if type(type_ids) is not torch.Tensor:
            type_ids = torch.tensor(type_ids)
        if type(y) is not torch.Tensor:
            y = torch.tensor(y)

        self.X = X
        self.mask = mask
        self.type_ids = type_ids
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.type_ids[idx], self.y[idx]


def recall_score(target, pred, average="micro", skip=-1):
    new_target = []
    new_pred = []
    for p, t in zip(pred, target):
        if t != skip:
            new_target.append(val_to_key(t, ner_dict))
            p = p if p != skip else 0
            new_pred.append(val_to_key(p, ner_dict))

    return metrics.recall_score([new_target], [new_pred], average=average)


def precision_score(target, pred, average="micro", skip=-1):
    new_target = []
    new_pred = []
    for p, t in zip(pred, target):
        if t != skip:
            new_target.append(val_to_key(t, ner_dict))
            p = p if p != skip else 0
            new_pred.append(val_to_key(p, ner_dict))

    return metrics.precision_score([new_target], [new_pred], average=average)


def f1_score(target, pred, skip=-1):
    new_target = []
    new_pred = []
    for p, t in zip(pred, target):
        if t != skip:
            new_target.append(val_to_key(t, ner_dict))
            p = p if p != skip else 0
            new_pred.append(val_to_key(p, ner_dict))

    return metrics.f1_score([new_target], [new_pred])


def get_texts_and_labels(dataset):
    tokens = dataset["tokens"]
    labels = dataset["ner_tags"]
    data = {
        "tokens": tokens,
        "labels": labels,
    }

    return data


def get_dataloader(data, batch_size, shuffle=True, drop_last=True):
    ids, mask, type_ids, labels = (
        data["input_ids"],
        data["attention_mask"],
        data["token_type_ids"],
        data["subword_labels"],
    )
    dataset = BertDataset(ids, mask, type_ids, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader


def key_to_val(key, dic):
    return dic[key] if key in dic else dic["UNK"]


def val_to_key(val, dic, pad_key="PAD"):
    keys = [k for k, v in dic.items() if v == val]
    if keys:
        return keys[0]
    return "UNK"


def path_to_data(path):
    with open(path, "r", encoding="utf-8") as f:
        row_data = f.readlines()

    data = []
    doc_index = []
    tokens = []
    labels = []
    replaced_labels = []
    pre_doc_end = 0
    for line in row_data:
        if "-DOCSTART-" in line:
            if len(tokens) != 0:
                labels = [key_to_val(la, ner_dict) for la in replaced_labels]
                document = dict(tokens=tokens, labels=labels, doc_index=doc_index)
                data.append(document)
                tokens = []
                replaced_labels = []
                labels = []
                doc_index = []
                pre_doc_end = 0

        elif len(line) <= 5:
            if len(labels) != 0:
                doc_start = pre_doc_end
                doc_end = len(tokens)
                doc_index.append((doc_start, doc_end))

                pre_doc_end = doc_end
                replaced_labels += boi1_to_2(labels)
                labels = []
        else:
            line = line.strip().split()

            tokens.append(line[0])
            labels.append(line[-1])

    if len(tokens) != 0:
        labels = [key_to_val(la, ner_dict) for la in replaced_labels]
        document = dict(tokens=tokens, labels=labels, doc_index=doc_index)
        data.append(document)
    return data
