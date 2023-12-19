import numpy as np
from sklearn.metrics import accuracy_score
from timm.scheduler import CosineLRScheduler
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.datamodule import BertDataset


def recall_score(target, pred, average=None, skip=-1):
    if type(target) not in [list, tuple]:
        target = target.tolist()
        pred = pred.tolist()
    target_type = list(set(target))
    count_dict = {t_t: [0, 0] for t_t in target_type if t_t != skip}
    for tar, pre in zip(target, pred):
        if tar == skip:
            continue
        if tar == pre:
            count_dict[tar][0] += 1
        else:
            count_dict[tar][1] += 1
    count_dict = [c_d[0] / sum(c_d) for c_d in count_dict.values()]
    if average is not None:
        count_dict = sum(count_dict) / len(count_dict)
    return np.array(count_dict)


def precision_score(target, pred, average=None, skip=-1):
    if type(target) not in [list, tuple]:
        target = target.tolist()
        pred = pred.tolist()
    pred_type = list(set(pred))
    count_dict = {p_t: [0, 0] for p_t in pred_type}
    for tar, pre in zip(target, pred):
        if tar == skip:
            continue
        if tar == pre:
            count_dict[pre][0] += 1
        else:
            count_dict[pre][1] += 1
    count_dict = [c_d[0] / sum(c_d) for c_d in count_dict.values() if sum(c_d) != 0]
    if average is not None:
        count_dict = sum(count_dict) / len(count_dict)
    return np.array(count_dict)


def f1_score(target, pred, skip=-1):
    recall = recall_score(target, pred, average="micro", skip=skip)
    prec = precision_score(target, pred, average="micro", skip=skip)

    return 2 / (1 / recall + 1 / prec)


class CosineScheduler(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, **kwargs):
        self.init_lr = optimizer.param_groups[0]["lr"]
        self.timmsteplr = CosineLRScheduler(optimizer, **kwargs)
        super().__init__(optimizer, self)

    def __call__(self, epoch):
        desired_lr = self.timmsteplr.get_epoch_values(epoch)[0]
        mult = desired_lr / self.init_lr
        return mult


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return acc


def get_texts_and_labels(dataset):
    tokens = dataset["tokens"]
    labels = dataset["ner_tags"]
    data = {
        "tokens": tokens,
        "labels": labels,
    }

    return data


def get_dataloader(data, batch_size, shuffle=True):
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
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


def key_to_val(key, dic):
    return dic[key] if key in dic else dic["UNK"]


def val_to_key(val, dic, pad_key="PAD"):
    keys = [k for k, v in dic.items() if v == val]
    if keys == pad_key:
        return "PAD"
    elif keys:
        return keys[0]
    return "UNK"


def path_to_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    data = data.split("\n\n")
    data = [d.split("\n") for d in data]

    text = []
    tag = []
    pos = []
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

    for sentence in tqdm(data):
        if len(sentence) == 0:
            continue

        divided = [s.split(" ") for s in sentence]
        text.append([d[0] for d in divided if len(d) == 4])
        tag.append([ner_dict[d[3]] for d in divided if len(d) == 4])
        pos.append(["0"] * len(tag[-1]))

    data = {
        "tokens": text,
        "ner_tags": tag,
        "pos_tags": pos,
    }
    return data