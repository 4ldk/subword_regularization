import copy
import os
import sys
from sklearn.utils import compute_sample_weight

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

subword_dict = {
    0: 0,
    1: 2,
    2: 2,
    3: 4,
    4: 4,
    5: 6,
    6: 6,
    7: 8,
    8: 8,
    9: 9,
}


class BertDataset(Dataset):
    def __init__(self, X, mask, y) -> None:
        super().__init__()
        if type(X) is not torch.Tensor:
            X = torch.tensor(X)
        if type(mask) is not torch.Tensor:
            mask = torch.tensor(mask)
        if type(y) is not torch.Tensor:
            y = torch.tensor(y)

        self.X = X
        self.mask = mask
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]


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
    ids, mask, labels = (
        data["input_ids"],
        data["attention_mask"],
        data["subword_labels"],
    )
    dataset = BertDataset(ids, mask, labels)
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


def get_label(word_ids, label, subword_label):
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(ner_dict["PAD"])
        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
            label_ids.append(label[word_idx])
        else:
            if subword_label == "I":
                label_ids.append(subword_dict[label[word_idx]])
            elif subword_label == "B":
                label_ids.append(label[word_idx])
            elif subword_label == "PAD":
                label_ids.append(ner_dict["PAD"])
            else:
                print("subword_label must be 'I', 'B' or 'PAD'.")
                exit(1)

        previous_word_idx = word_idx
    return label_ids


def dataset_encode(
    tokenizer,
    data,
    p=0,
    padding=512,
    return_tensor=True,
    subword_label="I",
    post_sentence_padding=False,
    add_sep_between_sentences=False,
):
    if p == 0:
        tokenizer.const_tokenize()
    else:
        tokenizer.random_tokenize()

    def max_with_none(lst):
        lst = [ls for ls in lst if ls is not None]
        return max(lst)

    def min_with_none(lst):
        lst = [ls for ls in lst if ls is not None]
        return min(lst)

    row_tokens = []
    row_labels = []
    input_ids = []
    attention_mask = []
    subword_labels = []
    predict_labels = []
    weight_y = []
    for document in data:
        text = document["tokens"]
        labels = document["labels"]
        max_length = padding - 2 if padding else len(text)

        for i, j in document["doc_index"]:
            subwords, word_ids = tokenizer.tokenzizeSentence(" ".join(text[i:j]))
            row_tokens.append(text[i:j])
            row_labels.append(labels[i:j])
            masked_ids = copy.deepcopy(word_ids)

            if post_sentence_padding:
                while len(subwords) < max_length and j < len(text):
                    if add_sep_between_sentences and j in [d[0] for d in document["doc_index"]]:
                        subwords.append(tokenizer.sep_token)
                        word_ids.append(None)
                        masked_ids.append(None)
                    ex_subwords = tokenizer.tokenize(" " + text[j])
                    subwords = subwords + ex_subwords
                    word_ids = word_ids + [max_with_none(word_ids) + 1] * len(ex_subwords)
                    masked_ids = masked_ids + [None] * len(ex_subwords)
                    j += 1
                    if len(subwords) < max_length:
                        subwords = subwords[:max_length]
                        word_ids = word_ids[:max_length]
                        masked_ids = masked_ids[:max_length]
            subwords = (
                [tokenizer.cls_token_id]
                + [tokenizer._convert_token_to_id(w) for w in subwords]
                + [tokenizer.sep_token_id]
            )
            word_ids = [None] + word_ids + [None]
            masked_ids = [None] + masked_ids + [None]

            if len(subwords) >= padding:
                subwords = subwords[:padding]
                word_ids = word_ids[:padding]
                masked_ids = masked_ids[:padding]
                mask = [1] * padding

            else:
                attention_len = len(subwords)
                pad_len = padding - len(subwords)
                subwords += [tokenizer.pad_token_id] * pad_len
                word_ids += [None] * pad_len
                masked_ids += [None] * pad_len
                mask = [1] * attention_len + [0] * pad_len

            input_ids.append(subwords)
            attention_mask.append(mask)

            label = labels[i:j]
            word_ids = [w_i - min_with_none(word_ids) if w_i is not None else None for w_i in word_ids]
            label_ids = get_label(word_ids, label, subword_label)
            subword_labels.append(label_ids)

            masked_label = row_labels[-1]
            masked_label_ids = get_label(masked_ids, masked_label, "PAD")
            predict_labels.append(masked_label_ids)

            weight_y += [l_i for l_i in label_ids if l_i != ner_dict["PAD"]]

    loss_rate = compute_sample_weight("balanced", y=weight_y)
    weight = [loss_rate[weight_y.index(i)] for i in range(len(set(weight_y)))]
    weight.append(0)

    if return_tensor:
        data = {
            "input_ids": torch.tensor(input_ids, dtype=torch.int),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.int),
            "subword_labels": torch.tensor(subword_labels, dtype=torch.long),
            "predict_labels": torch.tensor(predict_labels, dtype=torch.int),
            "tokens": row_tokens,
            "labels": row_labels,
            "weight": torch.tensor(weight, dtype=torch.float32),
        }
    else:
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "subword_labels": subword_labels,
            "predict_labels": predict_labels,
            "tokens": row_tokens,
            "labels": row_labels,
            "weight": weight,
        }
    return data


if __name__ == "__main__":
    from bpe_dropout import RobertaTokenizerDropout
    from tqdm import tqdm

    root_path = os.getcwd()
    tokenizer: RobertaTokenizerDropout = RobertaTokenizerDropout.from_pretrained("roberta-base", alpha=0.2)
    train_dataset = path_to_data(os.path.join(root_path, "test_datasets/eng.train"))
    for i in tqdm(range(3)):
        train_data = dataset_encode(
            tokenizer,
            train_dataset,
            p=0.1,
            padding=512,
            subword_label="PAD",
            post_sentence_padding=True,
            add_sep_between_sentences=True,
            return_tensor=False,
        )

        print(train_data["input_ids"][0])
