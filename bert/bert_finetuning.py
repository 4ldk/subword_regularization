import os
import random
import shutil
import sys

import hydra
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.maxMatchTokenizer import MaxMatchTokenizer
from utils.utils import CosineScheduler, f1_score, get_dataloader, path_to_data

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

model_dict = {
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


pos_dict = {
    '"': 0,
    "''": 1,
    "#": 2,
    "$": 3,
    "(": 4,
    ")": 5,
    ",": 6,
    ".": 7,
    ":": 8,
    "``": 9,
    "CC": 10,
    "CD": 11,
    "DT": 12,
    "EX": 13,
    "FW": 14,
    "IN": 15,
    "JJ": 16,
    "JJR": 17,
    "JJS": 18,
    "LS": 19,
    "MD": 20,
    "NN": 21,
    "NNP": 22,
    "NNPS": 23,
    "NNS": 24,
    "NN|SYM": 25,
    "PDT": 26,
    "POS": 27,
    "PRP": 28,
    "PRP$": 29,
    "RB": 30,
    "RBR": 31,
    "RBS": 32,
    "RP": 33,
    "SYM": 34,
    "TO": 35,
    "UH": 36,
    "VB": 37,
    "VBD": 38,
    "VBG": 39,
    "VBN": 40,
    "VBP": 41,
    "VBZ": 42,
    "WDT": 43,
    "WP": 44,
    "WP$": 45,
    "WRB": 46,
}


@hydra.main(config_path="../config", config_name="conll2003", version_base="1.1")
def main(cfg):
    os.environ["TRANSFORMERS_CACHE"] = "D:\\huggingface\\cashe"
    batch_size = cfg.batch_size
    lr = cfg.lr
    num_epoch = cfg.num_epoch
    warmup_t = cfg.warmup_t
    length = cfg.length
    p = cfg.p
    seed = cfg.seed
    model_name = cfg.model_name

    use_scheduler = cfg.use_scheduler
    lr_min = cfg.lr_min
    warmup_lr_init = cfg.warmup_lr_init

    train(
        batch_size,
        lr,
        num_epoch,
        warmup_t,
        length,
        p,
        seed,
        model_name,
        lr_min,
        warmup_lr_init,
        use_scheduler,
    )


def train(
    batch_size,
    lr,
    num_epoch,
    warmup_t,
    length,
    p,
    seed,
    model_name,
    lr_min,
    warmup_lr_init,
    use_scheduler=False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = "cuda"
    init_scale = 2048

    mmt = MaxMatchTokenizer(ner_dict=ner_dict, p=p, padding=length)
    bert_tokeninzer = AutoTokenizer.from_pretrained(model_name)
    mmt.loadBertTokenizer(bertTokenizer=bert_tokeninzer)

    train_dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/eng.train")
    valid_dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/eng.testa")
    valid_data = mmt.dataset_encode(valid_dataset, p=0)
    valid_loader = get_dataloader(valid_data, batch_size=batch_size, shuffle=True)

    print("Dataset Loaded")

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(ner_dict)).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        amsgrad=True,
    )
    scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)
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
        train_data = mmt.dataset_encode(train_dataset, p=p)
        train_loader = get_dataloader(train_data, batch_size=batch_size, shuffle=True)
        train_bar = tqdm(train_loader, leave=False, desc=f"Epoch{epoch} ")
        batch_loss = []
        for input, sub_input, label in train_bar:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                input, sub_input, label = (
                    input.to("cuda"),
                    sub_input.to("cuda"),
                    label.to("cuda"),
                )
                loss = model(input_ids=input, token_type_ids=None, attention_mask=sub_input, labels=label)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            optimizer.step()
            scaler.update()

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
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    input, sub_input, label = (
                        input.to("cuda"),
                        sub_input.to("cuda"),
                    )
                    pred = model(input, sub_input)
                pred = pred.squeeze(-1).to("cpu")
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
