import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import shutil
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
from torch import nn


class MyDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()

        self.inputs = dataset["texts"]
        self.y = torch.tensor(dataset["labels"], dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        y = self.y[idx]

        return item, y


def make_dataset(dataset, tokenizer, train=True):
    num = len(dataset["text"]) if train else int(len(dataset["text"]) * 0.3)
    texts = [d["text"] for i, d in enumerate(dataset) if i <= num]
    labels = [d["label"] for i, d in enumerate(dataset) if i <= num]
    new_dataset = {
        "texts": tokenizer(
            texts,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
            truncation=True,
        ).to("cuda"),
        "labels": labels,
    }
    return new_dataset


def predict():
    model_path = "./outputs/reranking/test_training/checkpoint-7500"
    model_name = "bert-base-cased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("imdb").shuffle(seed=42)
    dataset = MyDataset(make_dataset(dataset["test"], tokenizer, train=False))

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1, local_files_only=True)
    model = model.eval()
    print("Made model")

    preds = []
    probs = []
    labels = []
    with torch.inference_mode():
        for data in tqdm(dataset):
            label = data.pop("labels")
            data = {k: v.unsqueeze(0) for k, v in data.items()}
            # prob = model(**data)["logits"].reshape(-1).tolist()[0]
            # probs.append(prob)
            # preds.append(1 if prob >= 0.5 else 0)
            labels.append(label)
            print(label)

    print(probs[:30])
    print(accuracy_score(labels, preds))


def main():
    save_path = "./outputs/reranking/test_training"
    model_name = "bert-base-cased"  # "dslim/bert-large-NER" "Jean-Baptiste/roberta-large-ner-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("imdb").shuffle(seed=42)

    train_dataset = MyDataset(make_dataset(dataset["train"], tokenizer))
    train_loader = DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=True,
        drop_last=True,
    )
    eval_dataset = MyDataset(make_dataset(dataset["test"], tokenizer, train=False))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=10,
        shuffle=False,
        drop_last=True,
    )
    print("Made eval dataset")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, ignore_mismatched_sizes=True).to("cuda")
    print("Made model")

    optimizer = optim.Adam(
        model.parameters(),
        lr=3e-5,
        betas=(0.9, 0.999),
        amsgrad=True,
    )
    criteria = nn.BCEWithLogitsLoss()
    accs = []
    losses = []
    for epoch in range(3):
        model.train()
        train_bar = tqdm(train_loader, leave=False, desc=f"Epoch{epoch} ")
        batch_loss = []
        for inputs, label in train_bar:
            label = label.to("cuda")
            pred = model.forward(**inputs)["logits"]
            loss = criteria(pred.reshape(-1), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_postfix(loss=loss.to("cpu").item())
            batch_loss.append(loss.to("cpu").item())

        losses.append(sum(batch_loss) / len(batch_loss))

        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for inputs, label in tqdm(eval_loader, leave=False):
                pred = model.forward(**inputs)["logits"].to("cpu")
                pred = pred.squeeze(-1)
                preds.append(pred)
                labels.append(label)

            preds = torch.concatenate(preds).reshape(-1)
            labels = torch.concatenate(labels).reshape(-1)

            acc = accuracy_score(labels, (preds > 0.5).to(torch.int))
            accs.append(acc)

            print(f"Epoch{epoch}: train_loss: {sum(batch_loss) / len(batch_loss)}, acc: {acc}")

        s_path = f"{save_path}/epoch{epoch}.pth"
        torch.save(model.state_dict(), s_path)

        best_epoch = np.argmax(np.array(accs))
        print(f"best_epoch: {best_epoch}")
        shutil.copy(f"{save_path}/epoch{best_epoch}.pth", f"{save_path}/best.pth")


if __name__ == "__main__":
    main()
    # predict()
