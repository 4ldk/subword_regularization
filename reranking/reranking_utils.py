import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
import sys
import os
from datasets import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.utils import path_to_data


class RerankingDataset(Dataset):
    def __init__(self, dataset, device="cuda") -> None:
        super().__init__()
        self.sentences = dataset["Sentence"]
        self.replaced_sentence = dataset["Replaced_sentence"]
        self.predicted_label = dataset["Predicted_label"]
        self.golden_label = dataset["Golden_label"]
        self.inputs = dataset["inputs"]
        self.y = torch.tensor(dataset["y"], dtype=torch.float)
        self.device = device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item = {key: val[idx].to(self.device) for key, val in self.inputs.items()}
        y = self.y[idx]

        return item, y


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


def reranking_dataset(tokens, randoms, labels, consts=None):
    randoms = np.array(randoms).T.tolist()
    dataset = {
        "Sentence": [],
        "Sentence_id": [],
        "Replaced_sentence": [],
        "Predicted_label": [],
        "Golden_label": [],
        "y": [],
    }
    for rand in randoms:
        pred_pos = 0
        for sentence_id, token in enumerate(tokens):
            if len(token) == 0:
                continue
            divided_random = rand[pred_pos : pred_pos + len(token)]
            divided_label = labels[pred_pos : pred_pos + len(token)]

            acc = accuracy_score(divided_label, divided_random)
            y = max((acc - 0.8) * 5, 0)

            replaced_tokens = " ".join(
                [tok if d_rand == "O" else d_rand[2:] for tok, d_rand in zip(token, divided_random) if d_rand[0] != "I"]
            )

            check_dataset = [
                rep_sent
                for rep_sent, sent_id in zip(dataset["Replaced_sentence"], dataset["Sentence_id"])
                if sent_id == sentence_id
            ]
            if replaced_tokens not in check_dataset:
                dataset["Sentence"].append(token)
                dataset["Sentence_id"].append(sentence_id)
                dataset["Replaced_sentence"].append(replaced_tokens)
                dataset["Predicted_label"].append(divided_random)
                dataset["Golden_label"].append(divided_label)
                dataset["y"].append(y)

            pred_pos += len(token)

    if consts is not None:
        pred_pos = 0
        for sentence_id, token in enumerate(tokens):
            if len(token) == 0:
                continue
            divided_const = consts[pred_pos : pred_pos + len(token)]
            divided_label = labels[pred_pos : pred_pos + len(token)]

            acc = accuracy_score(divided_label, divided_const)
            y = max((acc - 0.8) * 5, 0)

            replaced_tokens = " ".join(
                [tok if d_rand == "O" else d_rand[2:] for tok, d_rand in zip(token, divided_const) if d_rand[0] != "I"]
            )

            check_dataset = [
                rep_sent
                for rep_sent, sent_id in zip(dataset["Replaced_sentence"], dataset["Sentence_id"])
                if sent_id == sentence_id
            ]
            if replaced_tokens not in check_dataset:
                dataset["Sentence"].append(token)
                dataset["Replaced_sentence"].append(replaced_tokens)
                dataset["Predicted_label"].append(divided_const)
                dataset["Golden_label"].append(divided_label)
                dataset["y"].append(y)
            pred_pos += len(token)
    return dataset


def get_dataset_from_100pred(path):
    consts = []
    randoms = []
    labels = []
    with open(path) as f:
        data = f.read().split("\n")
        for line in data:
            line = line.split(" ")
            labels.append(line[0])
            randoms.append(line[1:-1])
            consts.append(line[-1])
    labels = boi1_to_2(labels)
    return consts, randoms, labels


def sandwich_dataset(tokens, randoms, labels, consts=None):
    randoms = np.array(randoms).T.tolist()
    dataset = {
        "Sentence": [],
        "Sentence_id": [],
        "Replaced_sentence": [],
        "Predicted_label": [],
        "Golden_label": [],
        "y": [],
    }
    for rand in randoms:
        pred_pos = 0
        for sentence_id, token in enumerate(tokens):
            if len(token) == 0:
                continue
            divided_random = rand[pred_pos : pred_pos + len(token)]
            divided_label = labels[pred_pos : pred_pos + len(token)]

            acc = accuracy_score(divided_label, divided_random)
            y = max((acc - 0.8) * 5, 0)

            replaced_tokens = []
            pred_label = "O"
            for tok, d_rand in zip(token, divided_random):
                if d_rand[0] == "B":
                    replaced_tokens.append(f"<{d_rand[2:]}>")
                elif pred_label != "O" and d_rand == "O":
                    replaced_tokens.append(f"</{pred_label[2:]}>")
                replaced_tokens.append(tok)
                pred_label = d_rand

            replaced_tokens = " ".join(replaced_tokens)

            check_dataset = [
                rep_sent
                for rep_sent, sent_id in zip(dataset["Replaced_sentence"], dataset["Sentence_id"])
                if sent_id == sentence_id
            ]
            if replaced_tokens not in check_dataset:
                dataset["Sentence"].append(token)
                dataset["Sentence_id"].append(sentence_id)
                dataset["Replaced_sentence"].append(replaced_tokens)
                dataset["Predicted_label"].append(divided_random)
                dataset["Golden_label"].append(divided_label)
                dataset["y"].append(y)

            pred_pos += len(token)

    if consts is not None:
        pred_pos = 0
        for sentence_id, token in enumerate(tokens):
            if len(token) == 0:
                continue
            divided_const = consts[pred_pos : pred_pos + len(token)]
            divided_label = labels[pred_pos : pred_pos + len(token)]

            acc = accuracy_score(divided_label, divided_const)
            y = max((acc - 0.8) * 5, 0)

            replaced_tokens = []
            pred_label = "O"
            for tok, d_rand in zip(token, divided_random):
                if d_rand[0] == "B":
                    replaced_tokens.append(f"<{d_rand[2:]}>")
                elif pred_label != "O" and d_rand == "O":
                    replaced_tokens.append(f"</{pred_label[2:]}>")
                replaced_tokens.append(tok)
                pred_label = d_rand

            replaced_tokens = " ".join(replaced_tokens)
            check_dataset = [
                rep_sent
                for rep_sent, sent_id in zip(dataset["Replaced_sentence"], dataset["Sentence_id"])
                if sent_id == sentence_id
            ]
            if replaced_tokens not in check_dataset:
                dataset["Sentence"].append(token)
                dataset["Replaced_sentence"].append(replaced_tokens)
                dataset["Predicted_label"].append(divided_const)
                dataset["Golden_label"].append(divided_label)
                dataset["y"].append(y)
            pred_pos += len(token)
    return dataset


def get_dataset(dataset_names, tokenizer, sandwich=False):
    if "<PER>" not in tokenizer.vocab.keys():
        if sandwich:
            tokenizer.add_tokens(
                ["<PER>", "<ORG>", "<LOC>", "<MISC>", "</PER>", "</ORG>", "</LOC>", "</MISC>"],
                special_tokens=True,
            )
        else:
            tokenizer.add_tokens(
                ["<PER>", "<ORG>", "<LOC>", "<MISC>"],
                special_tokens=True,
            )

    if type(dataset_names) is list:
        dataset_list = [get_dataset(dataset_name, tokenizer)[0] for dataset_name in dataset_names]
        dataset = dataset_list[0]
        for k in dataset.keys():
            if type(dataset[k]) is list:
                for d in dataset_list[1:]:
                    dataset[k] += d[k]
            else:
                dataset[k] = torch.cat([d[k] for d in dataset_list], dim=0)
        return dataset, tokenizer

    consts, randoms, labels = get_dataset_from_100pred(f"./outputs100/output_{dataset_names}.txt")
    if dataset_names == "2023":
        dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/conll2023.txt")
    elif dataset_names == "valid":
        dataset = load_dataset("conll2003")["validation"]
    elif dataset_names == "test":
        dataset = load_dataset("conll2003")["test"]
    else:
        print("Unknown Dataset name")
        exit(1)

    if sandwich:
        dataset = sandwich_dataset(dataset["tokens"], randoms, labels, consts=consts)
    else:
        dataset = reranking_dataset(dataset["tokens"], randoms, labels, consts=consts)
    return dataset, tokenizer
