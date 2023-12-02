import os
import random
import sys
from tqdm.contrib import tzip

import numpy as np
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import seqeval.metrics
from reranking_utils import reranking_dataset, get_dataset_from_100pred

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
    model_path = "./outputs/reranking/model_base_ner/epoch2.pth"
    model_name = "dslim/bert-base-NER"

    consts, randoms, labels = get_dataset_from_100pred("./outputs100/output_test.txt")
    # dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/conll2023.txt")
    dataset = load_dataset("conll2003")["test"]
    dataset = reranking_dataset(dataset["tokens"], randoms, labels, consts=consts)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(["<PER>", "<ORG>", "<LOC>", "<MISC>"], special_tokens=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, ignore_mismatched_sizes=True)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(model_path))
    model = model.to("cuda")
    model = model.eval()
    print("Made model")

    pred = {}
    predicted_labels = []
    golden_labels = []
    with torch.inference_mode():
        for sentence_id, inputs, predicted_label, golden_label in tzip(
            dataset["Sentence_id"],
            dataset["Replaced_sentence"],
            dataset["Predicted_label"],
            dataset["Golden_label"],
        ):
            if sentence_id not in pred.keys():
                pred[sentence_id] = {
                    "Golden_label": golden_label,
                    "prob": [],
                    "Predicted_label": [],
                }
            inputs = tokenizer(inputs, return_tensors="pt").to("cuda")
            prob = model(**inputs)["logits"].reshape(-1).tolist()[0]
            pred[sentence_id]["prob"].append(prob)
            pred[sentence_id]["Predicted_label"].append(predicted_label)

    for p in pred.values():
        max_prob_index = np.argmax(np.array(p["prob"]))
        final_pred = p["Predicted_label"][max_prob_index]
        predicted_labels += final_pred
        golden_labels += p["Golden_label"]

    print(seqeval.metrics.classification_report([golden_labels], [predicted_labels], digits=4))


if __name__ == "__main__":
    main()
