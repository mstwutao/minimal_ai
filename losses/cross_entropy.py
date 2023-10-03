import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax(x):
    return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)

def nl(inputs, targets):
    return -inputs[range(targets.shape[0]), targets].log().mean()


# While mathematically equivalent to log(softmax(x)), doing these two operations separately is slower, and numerically unstable. 
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def nll(inputs, targets):
    return -inputs[range(targets.shape[0]), targets].mean()


if __name__ == "__main__":
    batch_size, n_classes = 32, 10
    x = torch.randn(batch_size, n_classes)
    target = torch.randint(n_classes, size=(batch_size, ), dtype=torch.long)

    # The following 4 ways are equal
    
    pred = softmax(x)
    loss=nl(pred, target)
    print(loss)

    pred = log_softmax(x)
    loss = nll(pred, target)
    print(loss)

    pred = F.log_softmax(x, dim=-1)
    loss = F.nll_loss(pred, target)
    print(loss)

    loss = F.cross_entropy(x, target)
    print(loss)
