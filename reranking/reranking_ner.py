import os
import random
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from reranking_utils import (
    RerankingDataset,
    reranking_dataset,
    get_dataset_from_100pred,
)
from torch import optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.utils import path_to_data

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
tags = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}


def main():
    save_path = "./outputs/reranking/model_base_ner"
    model_name = "dslim/bert-base-NER"  # "Jean-Baptiste/roberta-large-ner-english" "dslim/bert-large-NER"
    lr = 3e-5
    batch_size = 10
    epochs = 10

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(["PER", "ORG", "LOC", "MISC"], special_tokens=True)

    consts, randoms, labels = get_dataset_from_100pred("./outputs100/output_2023.txt")
    dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/conll2023.txt")
    dataset = reranking_dataset(dataset["tokens"], randoms, labels, consts=consts)
    dataset["inputs"] = tokenizer(
        dataset["Replaced_sentence"], padding="max_length", max_length=512, return_tensors="pt"
    ).to("cuda")
    train_dataset = RerankingDataset(dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    print("Made train dataset")

    consts, randoms, labels = get_dataset_from_100pred("./outputs100/output_valid.txt")
    dataset = load_dataset("conll2003")["validation"]
    dataset = reranking_dataset(dataset["tokens"], randoms, labels, consts=consts)
    dataset["inputs"] = tokenizer(
        dataset["Replaced_sentence"], padding="max_length", max_length=512, return_tensors="pt"
    ).to("cuda")
    eval_dataset = RerankingDataset(dataset)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    print("Made eval dataset")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, ignore_mismatched_sizes=True)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to("cuda")

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        amsgrad=True,
    )
    criteria = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(train_loader, leave=False, desc=f"Epoch{epoch} ")
        batch_loss = []
        for inputs, label in train_bar:
            label = label.to("cuda")
            pred = model.forward(**inputs)["logits"]
            pred = nn.functional.sigmoid(pred)
            loss = criteria(pred.reshape(-1), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_postfix(loss=loss.to("cpu").item())
            batch_loss.append(loss.to("cpu").item())

        losses.append(sum(batch_loss) / len(batch_loss))

        model.eval()
        test_losses = []
        with torch.no_grad():
            for inputs, label in tqdm(eval_loader, leave=False):
                label = label.to("cuda")
                pred = model.forward(**inputs)["logits"]
                pred = nn.functional.sigmoid(pred)
                test_losses.append(criteria(pred.reshape(-1), label).to("cpu").item())

            print(
                f"Epoch{epoch}: train_loss: {sum(batch_loss) / len(batch_loss)}, test_loss: {sum(test_losses) / len(test_losses)}"
            )

        s_path = f"{save_path}/epoch{epoch}.pth"
        torch.save(model.state_dict(), s_path)


if __name__ == "__main__":
    main()
