import os
import sys

import numpy as np
import seqeval.metrics
from datasets import load_dataset
from reranking_utils import get_dataset_from_100pred, reranking_dataset
from sklearn.metrics import accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.utils import path_to_data


def main():
    consts, randoms, labels = get_dataset_from_100pred("./outputs100/output_2023.txt")
    dataset = path_to_data("C:/Users/chenr/Desktop/python/subword_regularization/test_datasets/conll2023.txt")
    # dataset = load_dataset("conll2003")["test"]
    dataset = reranking_dataset(dataset["tokens"], randoms, labels, consts=consts)

    pred = {}
    predicted_labels = []
    golden_labels = []

    for sentence_id, inputs, predicted_label, golden_label in zip(
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
        pred[sentence_id]["Predicted_label"].append(predicted_label)
        pred[sentence_id]["prob"].append(accuracy_score(golden_label, predicted_label))

    for p in pred.values():
        max_prob_index = np.argmax(np.array(p["prob"]))
        final_pred = p["Predicted_label"][max_prob_index]
        predicted_labels += final_pred
        golden_labels += p["Golden_label"]

    print(seqeval.metrics.classification_report([golden_labels], [predicted_labels], digits=4))


if __name__ == "__main__":
    main()
