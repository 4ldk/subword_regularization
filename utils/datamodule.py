import torch
from torch import nn
from torch.utils.data import Dataset
from TorchCRF import CRF


class BertCRF(nn.Module):
    def __init__(self, model, batch_size, num_labels, length, device="cuda") -> None:
        super().__init__()
        self.model = model
        self.crf = CRF(num_labels).to(device)
        self.mask = torch.ones(batch_size, length).to(torch.bool).to(device)

    def forward(self, x, p, y):
        preds = self.model(x, p)
        if type(preds) is torch.Tensor:
            x = preds
        else:
            x = preds["logits"]
        out = self.crf.forward(x, y, self.mask)

        return out

    def decode(self, x, p):
        preds = self.model(x, p)
        if type(preds) is torch.Tensor:
            x = preds
        else:
            x = preds["logits"]
        out = self.crf.viterbi_decode(x, self.mask)
        out = torch.tensor(out)

        return out


class BertDataset(Dataset):
    def __init__(self, X, mask, y) -> None:
        super().__init__()
        if type(X) is not torch.Tensor:
            X = torch.tensor(X)
        if type(mask) is torch.Tensor:
            mask = torch.tensor(mask)
        if type(y) is not torch.Tensor:
            y = torch.tensor(y)

        self.X = X
        self.mask = mask
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.mask[idx], self.y[idx]
