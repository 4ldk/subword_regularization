import pickle

import numpy as np
import seqeval.metrics
import matplotlib.pyplot as plt


def main():
    use_datasets = ["test", "valid", "2023"]  # ["test","2023","valid"]
    max_num = 100
    dataset_list = []

    for u_d in use_datasets:
        with open(f"./output100/dataset_{u_d}.pkl", mode="wb") as f:
            dataset_list.append(pickle.load(f))

    dataset = dataset_list[0]

    sentence_id_start = max(dataset["Sentence_id"]) + 1
    for k in dataset.keys():
        if type(dataset[k]) is list:
            for d in dataset_list[1:]:
                if k == "Sentence_id":
                    d[k] = [id + sentence_id_start for id in d[k]]
                    sentence_id_start = max(d[k]) + 1
                dataset[k] += d[k]
    pred = {}

    f1_for_length = []
    length = []
    for sentence_id, golden_label in zip(
        dataset["Sentence_id"],
        dataset["Golden_label"],
    ):
        if sentence_id not in pred.keys():
            f1 = seqeval.metrics.f1_score([golden_label], [dataset["consts"][sentence_id]], zero_division=0)
            pred[sentence_id] = [f1, 0]
            f1_for_length.append(f1)
            length.append(len(golden_label))
        pred[sentence_id][1] += 1

    pred = list(pred.values())
    pred = sorted(pred, key=lambda x: x[1])
    pre_num = 0
    f1s = []
    ave_f1s = []
    nums = []
    counts = []
    for p in pred:
        if p[1] != pre_num and len(f1s) != 0:
            nums.append(pre_num)
            counts.append(len(f1s))
            ave_f1s.append(sum(f1s) / len(f1s))
            f1s = []

        f1s.append(p[0])
        pre_num = p[1]

    nums.append(pre_num)
    counts.append(len(f1s))
    ave_f1s.append(sum(f1s) / len(f1s))

    pred = np.array(pred)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.bar(nums[:max_num], counts[:max_num])
    ax2.plot(nums[:max_num], ave_f1s[:max_num], color="k")
    ax2.set_ylim(0.75, 1)
    ax.set_yscale("log")

    plt.show()

    plt.scatter(length, f1_for_length)
    plt.show()


if __name__ == "__main__":
    main()
