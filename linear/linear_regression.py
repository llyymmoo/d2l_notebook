# liuyumin 2023 0524

import torch
import numpy as np
from IPython import display
from matplotlib import pyplot as plt
import random

# ----------------------------------------------------------------
# self defind functions

def fig_display(figsize=(3.5, 2.5)):
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize

def data_iter(batch_size, features, labels):
    if (features.size(0) != labels.size(0)):
        return -1
    
    num_samples = len(features)
    indexs = list(range(num_samples))
    random.shuffle(indexs)
    for i in range(0, num_samples, batch_size):
        j = torch.LongTensor(indexs[i : min(i+batch_size, num_samples)])
        yield features.index_select(0, j), labels.index_select(0, j)

def linreg(X, w, b):
    return torch.mm(X, w) + b

def square_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param.data += -lr * param.grad / batch_size

# self defind function end
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# main function

# generate datas
num_input = 2
num_samples = 1000
w_t = torch.tensor([2, -3.4]).view(2, 1)
b_t = torch.tensor([4.2])
features = torch.randn(num_samples, num_input, dtype=torch.float32)
labels = torch.mm(features, w_t) + b_t
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()))

# visualise datas
# fig_display()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()

# train model
w = torch.tensor(np.random.normal(0, 0.01, (num_input, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

lr = 0.03
batch_size = 10
num_epochs = 5
net = linreg
loss = square_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
    
    train_loss = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_loss.mean().item()))

print(w)
print(b)