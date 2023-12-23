import torch
from seqeval import metrics

a = torch.tensor([0, 0, 0, 1, 2, 2, 1, 1, 2])
b = torch.tensor([0, 0, 0, 1, 2, 1, 1, 1, 2])

dic = {"O": 0, "B-LOC": 1, "I-LOC": 2}
a = [[k for k, v in dic.items() if v == aa][0] for aa in a]
b = [[k for k, v in dic.items() if v == aa][0] for aa in b]

print(metrics.f1_score([a], [b]))
