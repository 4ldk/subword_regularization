import os
import random
import sys
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from reranking_utils import (
    RerankingDataset,
    sandwich_dataset,
    get_dataset_from_100pred,
)

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
    save_path = "./outputs/reranking/sand_model_base_normal"
    model_name = "bert-base-cased"  # "dslim/bert-large-NER" "Jean-Baptiste/roberta-large-ner-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(
        ["<PER>", "<ORG>", "<LOC>", "<MISC>", "</PER>", "</ORG>", "</LOC>", "</MISC>"],
        special_tokens=True,
    )

    consts, randoms, labels = get_dataset_from_100pred("./outputs100/output_2023.txt")
    dataset = path_to_data(
        "C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/conll2023.txt"
    )
    dataset = sandwich_dataset(dataset["tokens"], randoms, labels, consts=consts)
    dataset["inputs"] = tokenizer.batch_encode_plus(
        dataset["Replaced_sentence"],
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    train_dataset = RerankingDataset(dataset)
    print("Made train dataset")

    consts, randoms, labels = get_dataset_from_100pred("./outputs100/output_valid.txt")
    dataset = load_dataset("conll2003")["validation"]
    dataset = sandwich_dataset(dataset["tokens"], randoms, labels, consts=consts)
    dataset["inputs"] = tokenizer.batch_encode_plus(
        dataset["Replaced_sentence"],
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    eval_dataset = RerankingDataset(dataset)
    print("Made eval dataset")

    training_args = TrainingArguments(
        output_dir=save_path,
        learning_rate=5e-3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=20,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    print("Made model")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    model.save_pretrained(save_path)


if __name__ == "__main__":
    main()
