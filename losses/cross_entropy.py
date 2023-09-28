import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax(x):
    return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)


def nl(input, target):
    return -input[range(target.shape[0]), target].log().mean()


if __name__ == "__main__":
    batch_size, n_classes = 32, 10
    x = torch.randn(batch_size, n_classes)
    target = torch.randint(n_classes, size=(batch_size,), dtype=torch.long)
