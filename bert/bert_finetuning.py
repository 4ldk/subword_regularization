import os
import random
import shutil
import sys

import hydra
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.maxMatchTokenizer import MaxMatchTokenizer
from utils.utils import f1_score, get_dataloader, path_to_data

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


@hydra.main(config_path="../config", config_name="conll2003", version_base="1.1")
def main(cfg):
    os.environ["TRANSFORMERS_CACHE"] = "D:\\huggingface\\cashe"
    batch_size = cfg.batch_size
    lr = cfg.lr
    num_epoch = cfg.num_epoch
    length = cfg.length
    p = cfg.p
    seed = cfg.seed
    model_name = cfg.model_name

    accum_iter = cfg.accum_iter
    weight_decay = cfg.weight_decay
    use_loss_weight = cfg.use_loss_weight
    use_scheduler = cfg.use_scheduler
    warmup_late = cfg.warmup_late

    train(
        batch_size,
        lr,
        num_epoch,
        length,
        p,
        seed,
        model_name,
        accum_iter=accum_iter,
        weight_decay=weight_decay,
        use_loss_weight=use_loss_weight,
        use_scheduler=use_scheduler,
        warmup_late=warmup_late,
    )


def train(
    batch_size,
    lr,
    num_epoch,
    length,
    p,
    seed,
    model_name,
    accum_iter=4,
    weight_decay=0,
    use_loss_weight=False,
    use_scheduler=False,
    warmup_late=0.01,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = "cuda"
    init_scale = 4096
    os.makedirs("./model")

    mmt = MaxMatchTokenizer(ner_dict=ner_dict, p=p, padding=length)
    bert_tokeninzer = AutoTokenizer.from_pretrained(model_name)
    mmt.loadBertTokenizer(bertTokenizer=bert_tokeninzer)

    train_dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/eng.train")
    train_data = mmt.dataset_encode(train_dataset, p=p, subword_label="PAD")
    train_loader = get_dataloader(train_data, batch_size=batch_size, shuffle=True)

    valid_dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/eng.testa")
    valid_data = mmt.dataset_encode(valid_dataset, p=0, subword_label="PAD")
    valid_loader = get_dataloader(valid_data, batch_size=batch_size, shuffle=True)

    test_dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/eng.testb")
    test_data = mmt.dataset_encode(test_dataset, p=0, subword_label="PAD")
    test_loader = get_dataloader(test_data, batch_size=batch_size, shuffle=True)

    print("Dataset Loaded")

    weight = train_data["weight"].to(device) if use_loss_weight else None
    num_training_steps = int(len(train_loader) / accum_iter) * num_epoch
    num_warmup_steps = int(num_training_steps * warmup_late)

    net = model_train(
        model_name,
        device,
        lr,
        accum_iter,
        weight_decay,
        weight,
        num_warmup_steps,
        num_training_steps,
        use_scheduler,
        init_scale,
    )

    f1s = []
    losses = []
    for epoch in tqdm(range(num_epoch)):
        loss, _, _ = net.step(epoch, train_loader, train=True)
        _, valid_acc, valid_f1 = net.step(epoch, valid_loader, train=False)
        _, _, test_f1 = net.step(epoch, test_loader, train=False)

        f1s.append(valid_f1)
        losses.append(loss)
        tqdm.write(
            f"Epoch{epoch}: loss: {loss}, val_acc: {valid_acc:.4f}, val_f1: {valid_f1:.4f}, tes_f1: {test_f1:.4f}"
        )

        if epoch != num_epoch - 1 and p != 0:
            train_data = mmt.dataset_encode(train_dataset, p=p)
            train_loader = get_dataloader(train_data, batch_size=batch_size, shuffle=True)

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


class model_train:
    def __init__(
        self,
        model_name,
        device,
        lr,
        accum_iter,
        weight_decay,
        weight,
        num_warmup_steps,
        num_training_steps,
        use_scheduler,
        init_scale,
    ):
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(ner_dict)).to(device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            amsgrad=True,
        )
        self.loss_func = nn.CrossEntropyLoss(weight=weight, ignore_index=ner_dict["PAD"])
        self.scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)
        if use_scheduler:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)

        self.device = device
        self.use_scheduler = use_scheduler
        self.accum_iter = accum_iter

    def step(self, epoch, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()
        bar = tqdm(loader, leave=False, desc=f"Epoch{epoch} ")
        batch_loss, preds, labels = [], [], []
        final_loss, acc, f1 = 0, 0, 0

        for batch_idx, (input, sub_input, label) in enumerate(bar):
            input, sub_input, label = (
                input.to(self.device),
                sub_input.to(self.device),
                label.to(self.device),
            )

            if train:
                with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                    logits, pred, loss = self.forward(input, sub_input, label)

                self.scaler.scale(loss / self.accum_iter).backward()
                if ((batch_idx + 1) % self.accum_iter == 0) or (batch_idx + 1 == len(loader)):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.optimizer.step()
                    self.scaler.update()

                    if self.use_scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                batch_loss.append(loss.to("cpu").item())
            else:
                with torch.no_grad():
                    with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                        logits, pred, loss = self.forward(input, sub_input, label)
                    preds.append(pred)
                    labels.append(label.to("cpu"))

            bar.set_postfix(loss=loss.to("cpu").item())

        if train:
            final_loss = sum(batch_loss) / len(batch_loss)
            save_path = f"./model/epoch{epoch}.pth"
            torch.save(self.model.state_dict(), save_path)
        else:
            acc, f1 = self.get_score(preds, labels)
        return final_loss, acc, f1

    def get_score(self, preds, labels):
        preds = torch.concatenate(preds)
        labels = torch.concatenate(labels)

        pad = torch.logical_not((torch.ones_like(labels) * ner_dict["PAD"] == labels))
        data_num = ((labels == labels) * pad).sum()

        acc = ((preds == labels) * pad).sum().item() / data_num

        preds = preds.view(-1)
        labels = labels.view(-1)
        f1 = f1_score(labels, preds, skip=ner_dict["PAD"]).tolist()

        return acc, f1

    def forward(self, input, sub_input, label):
        logits = self.model(input, sub_input).logits
        pred = logits.squeeze(-1).argmax(-1)

        if label is not None:
            loss = self.loss_func(logits.view(-1, len(ner_dict)), label.view(-1))
            return logits, pred.to("cpu"), loss

        return logits, pred.to("cpu")


if __name__ == "__main__":
    main()
