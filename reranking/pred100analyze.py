import pickle

import numpy as np
import seqeval.metrics
import matplotlib.pyplot as plt


def main():
    use_datasets = ["test"]  # ["test","2023","valid"]
    max_num = 100
    zero_division = "skip"  # 0, 1, skip
    dataset_list = []

    for u_d in use_datasets:
        with open(f"./outputs100/dataset_{u_d}.pkl", mode="rb") as f:
            dataset = pickle.load(f)
        dataset_list.append(dataset)

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
    skip_num = 0
    for sentence_id, golden_label in zip(
        dataset["Sentence_id"],
        dataset["Golden_label"],
    ):
        if sentence_id not in pred.keys():
            if (
                zero_division == "skip"
                and len(set(golden_label)) == 1
                and golden_label[0] == "O"
                and len(set(dataset["consts"][sentence_id])) == 1
                and dataset["consts"][sentence_id][0] == "O"
            ):
                skip_num += 1
                continue
            f1 = seqeval.metrics.f1_score([golden_label], [dataset["consts"][sentence_id]], zero_division=zero_division)
            pred[sentence_id] = [f1, 0]
            f1_for_length.append(f1)
            length.append(len(golden_label))
        pred[sentence_id][1] += 1

    print("skip num is ", skip_num)
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

    length = np.array(length)
    f1_for_length = np.array(f1_for_length)
    plt.scatter(length, f1_for_length, s=4)
    a, b = np.polyfit(length, f1_for_length, 1)
    # フィッティング直線
    y2 = a * length + b
    plt.plot(length, y2, color="k")
    plt.show()


if __name__ == "__main__":
    main()
