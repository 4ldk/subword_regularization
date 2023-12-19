import seqeval.metrics
from LUKE_utils import round_preds


if __name__ == "__main__":
    path = "C:/Users/chenr/Desktop/python/subword_regularization/outputs/bert_pred_roop/2023-12-19/bertner_large_03_test/many_preds.txt"
    consts = []
    randoms = []
    labels = []
    with open(path) as f:
        data = f.read().split("\n")
        for line in data:
            if len(line) == 0:
                continue
            line = line.split(" ")
            labels.append(line[2])
            randoms.append(line[3:-1])
            consts.append(line[-1])
    pred_func = round_preds()
    round_prediction = [pred_func.majority([co]) for ra, co, la in zip(randoms, consts, labels)]
    print(seqeval.metrics.classification_report([labels], [round_prediction], digits=4))
