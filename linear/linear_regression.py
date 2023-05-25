# liuyumin 2023 0524

import torch
from torch import nn
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

# ----------------------------------------------------------------
# from scratch
# train model
# w = torch.tensor(np.random.normal(0, 0.01, (num_input, 1)), dtype=torch.float32)
# b = torch.zeros(1, dtype=torch.float32)
# w.requires_grad_(requires_grad=True)
# b.requires_grad_(requires_grad=True)

# lr = 0.03
# batch_size = 10
# num_epochs = 5
# net = linreg
# loss = square_loss

# for epoch in range(num_epochs):
#     for X, y in data_iter(batch_size, features, labels):
#         l = loss(net(X, w, b), y).sum()
#         l.backward()
#         sgd([w, b], lr, batch_size)

#         w.grad.data.zero_()
#         b.grad.data.zero_()
    
#     train_loss = loss(net(features, w, b), labels)
#     print('epoch %d, loss %f' % (epoch + 1, train_loss.mean().item()))

# print(w)
# print(b)

# # concise version
class LinearRegression(nn.Module):
    def __init__(self, num_outputs):
        super(LinearRegression, self).__init__()
        self.net = nn.LazyLinear(num_outputs)
    
    def forward(self, X):
        return self.net(X)

device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 10
lr = 0.03
batch_size = 16

valid_inputs = features.to(device)
valid_labels = labels.to(device)

model = LinearRegression(1).to(device)
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        l = loss(outputs, y)
        l.backward()
        optimizer.step()
    
    with torch.no_grad():
        predict = model(valid_inputs)
        train_loss = loss(predict, valid_labels)
    print('epoch %d, loss %f' % (epoch + 1, train_loss.mean().item()))

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")