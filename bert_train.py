import os
import random
import shutil

import hydra
import numpy as np
import torch
from datasets import load_dataset
from torch import optim
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from utils.boi_convert import boi2_to_1
from utils.datamodule import BertCRF
from utils.maxMatchTokenizer import MaxMatchTokenizer
from utils.utils import CosineScheduler, f1_score, get_dataloader, get_texts_and_labels


@hydra.main(config_path="./config", config_name="conll2003", version_base="1.1")
def main(cfg):
    batch_size = cfg.batch_size
    lr = cfg.lr
    num_epoch = cfg.num_epoch
    warmup_t = cfg.warmup_t
    length = cfg.length
    p = cfg.p
    seed = cfg.seed
    device = cfg.device
    model_name = cfg.model_name

    use_scheduler = cfg.use_scheduler
    lr_min = cfg.lr_min
    warmup_lr_init = cfg.warmup_lr_init

    stop_word = cfg.stop_word
    as_aug = cfg.as_aug
    boi1 = cfg.boi1

    train(
        batch_size,
        lr,
        num_epoch,
        warmup_t,
        length,
        p,
        seed,
        device,
        model_name,
        lr_min,
        warmup_lr_init,
        use_scheduler,
        stop_word,
        as_aug,
        boi1,
    )


def train(
    batch_size,
    lr,
    num_epoch,
    warmup_t,
    length,
    p,
    seed,
    device,
    model_name,
    lr_min,
    warmup_lr_init,
    use_scheduler=False,
    stop_word=False,
    as_aug=False,
    boi1=False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    os.makedirs("./model")

    dataset = load_dataset("conll2003")
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]

    if boi1:
        train_dataset = boi2_to_1(train_dataset)
        valid_dataset = boi2_to_1(valid_dataset)

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

    mmt = MaxMatchTokenizer(ner_dict=ner_dict, p=p, padding=length, stop_word=stop_word)
    bert_tokeninzer = AutoTokenizer.from_pretrained(model_name)
    mmt.loadBertTokenizer(bertTokenizer=bert_tokeninzer)

    train_data = get_texts_and_labels(train_dataset)
    valid_data = get_texts_and_labels(valid_dataset)
    valid_data = mmt.dataset_encode(valid_data, p=0)
    valid_loader = get_dataloader(valid_data, batch_size=batch_size, shuffle=True)

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(ner_dict))
    model = BertCRF(model, batch_size, len(ner_dict), length).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        amsgrad=True,
    )
    if use_scheduler:
        scheduler = CosineScheduler(
            optimizer,
            t_initial=num_epoch - warmup_t,
            lr_min=lr_min,
            warmup_t=warmup_t,
            warmup_lr_init=warmup_lr_init,
            warmup_prefix=True,
        )

    f1s = []
    losses = []
    for epoch in range(num_epoch):
        model.train()
        train_data = mmt.dataset_encode(train_data, p=p)
        train_loader = get_dataloader(train_data, batch_size=batch_size, shuffle=True)
        train_bar = tqdm(train_loader, leave=False, desc=f"Epoch{epoch} ")
        batch_loss = []
        for input, sub_input, label in train_bar:
            input, sub_input, label = (
                input.to(device),
                sub_input.to(device),
                label.to(device),
            )
            loss = model.forward(input, sub_input, label)
            loss = torch.sum(-loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_postfix(loss=loss.to("cpu").item())
            batch_loss.append(loss.to("cpu").item())

        if as_aug:
            train_data = mmt.dataset_encode(train_data, p=0)
            train_loader = get_dataloader(train_data, batch_size=batch_size, shuffle=True)
            train_bar = tqdm(train_loader, leave=False, desc=f"Epoch{epoch}_2 ")
            for input, sub_input, label in train_bar:
                input, sub_input, label = (
                    input.to(device),
                    sub_input.to(device),
                    label.to(device),
                )
                loss = -model.forward(input, sub_input, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_bar.set_postfix(loss=loss.to("cpu").item())
                batch_loss.append(loss.to("cpu").item())

        losses.append(sum(batch_loss) / len(batch_loss))

        if use_scheduler:
            scheduler.step()

        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for input, sub_input, label in tqdm(valid_loader, leave=False):
                input, sub_input = (
                    input.to(device),
                    sub_input.to(device),
                )

                pred = model.decode(input, sub_input).to("cpu")
                pred = pred.squeeze(-1)
                preds.append(pred)
                labels.append(label)

            preds = torch.concatenate(preds)
            labels = torch.concatenate(labels)
            pad = torch.logical_not((torch.ones_like(labels) * ner_dict["PAD"] == labels))
            data_num = ((labels == labels) * pad).sum()
            acc = ((preds == labels) * pad).sum().item() / data_num
            preds = preds.view(-1)
            labels = labels.view(-1)
            f1 = f1_score(labels, preds, skip=ner_dict["PAD"]).tolist()
            f1s.append(f1)

            print(f"Epoch{epoch}: train_loss: {sum(batch_loss) / len(batch_loss)}, acc: {acc}, f1: {f1}")

        save_path = f"./model/epoch{epoch}.pth"
        torch.save(model.state_dict(), save_path)

    with open("./train_valid_score.csv", "w") as f:
        min_loss = 100
        max_f1 = 0
        best_epoch = 0
        for i, (loss, f1) in enumerate(zip(losses, f1s)):
            f.write(f"{i}, {loss}, {f1}\n")
            if f1 > max_f1:
                max_f1 = f1
                best_epoch = i
            elif f1 == max_f1:
                if loss < min_loss:
                    min_loss = loss
                    best_epoch = i

        print(f"best_epoch: {best_epoch}")
        shutil.copy(f"./model/epoch{best_epoch}.pth", "./model/best.pth")


if __name__ == "__main__":

    main()
