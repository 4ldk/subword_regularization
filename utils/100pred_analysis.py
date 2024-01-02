import matplotlib.pyplot as plt
import numpy as np
import seqeval.metrics


def main():
    use_datasets = ["test", "2023", "valid"]  # ["test","2023","valid"]
    model_type = "BertB"  # BertL, RobertaL, BertB, RobertaB
    model = "Normal"  # Reg, Normal, Reg3
    max_num = 10
    zero_division = "skip"  # "skip"  # 0, 1, skip

    datasets = []

    for u_d in use_datasets:
        if u_d == 2003:
            encoding = "cp-932"
        else:
            encoding = "utf-8"
        if model_type == "BertL":
            path = f"./outputs100/{model_type}/{model}{u_d}/many_preds.txt"
        else:
            path = f"./outputs100/{model_type}/{model}{u_d}.txt"
        with open(path, encoding=encoding) as f:
            dataset = f.read()
        dataset = dataset.split("\n\n")
        datasets += dataset

    pred = []
    f1_for_length = []
    length = []
    skip_num = 0
    for d_s in datasets:
        if len(d_s) == 0:
            continue
        golden_label = []
        preds = []
        consts = []
        for i, d in enumerate(d_s.split("\n")):
            d = d.split()
            golden_label.append(d[1])
            consts.append(d[-1])
            for j, dd in enumerate(d[2:]):
                if i == 0:
                    preds.append([dd])
                else:
                    preds[j].append(dd)

        preds = set([" ".join(p) for p in preds])

        if (
            zero_division == "skip"
            and len(set(golden_label)) == 1
            and golden_label[0] == "O"
            and len(set(consts)) == 1
            and consts[0] == "O"
        ):
            skip_num += 1
            continue
        f1 = seqeval.metrics.f1_score([golden_label], [consts], zero_division=zero_division)
        pred.append([f1, len(preds)])
        f1_for_length.append(f1)
        length.append(len(golden_label))

    print("skip num is ", skip_num)
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

    print(f"uniformed Answer rate: {counts[0]/sum(counts)}")
    pred = np.array(pred)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.bar(nums[:max_num], counts[:max_num])
    ax.plot(nums[:max_num], ave_f1s[:max_num], color="k")
    ax2.set_yscale("log")
    ax.set_ylim(0.70, 1)
    ax2.set_ylim(0.9, 7500)

    ax.set_xlabel("Number of output variations")
    ax.set_ylabel("F1 micro average")
    ax2.set_ylabel("Number of sentences")

    ax.set_zorder(2)
    ax2.set_zorder(1)

    ax.patch.set_alpha(0)
    plt.show()

    # """
    # ラベル数-F11の分布表示
    length = np.array(length)
    f1_for_length = np.array(f1_for_length)
    plt.scatter(length, f1_for_length, s=4)
    a, b = np.polyfit(length, f1_for_length, 1)
    # フィッティング直線
    y2 = a * length + b
    plt.plot(length, y2, color="k")

    plt.xlabel("Number of labels")
    plt.ylabel("F1 micro average")

    plt.show()
    # """


if __name__ == "__main__":
    main()
