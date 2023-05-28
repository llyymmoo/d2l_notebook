import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

class Classifier(nn.Module):
    def __init__(self, num_outputs):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_outputs)
        )
    
    def forward(self, x):
        return self.net(x)
# load data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# general settings
device = "cuda" if torch.cuda.is_available() else "cpu"
num_outputs = 10
batch_size = 64
lr = 0.05
num_epoches = 50

training_dataloader = DataLoader(training_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

# nn definition
model = Classifier(num_outputs).to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

# ---------------------------------------------------------------------
# attention!!!
# in pytorch, fuction nn.CrossEntropyLoss(), nn.NLLLoss(), nn.BCELoss()
# is quite confused, it is recommended to read the implementation code &
# do some test
# ---------------------------------------------------------------------

# train
for i in range(num_epoches):
    for x, y in training_dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x)
        l = loss(output, y)
        l.backward()
        optimizer.step()

    with torch.no_grad():
        error = 0.0
        num_test = 0
        for x_t, y_t in test_dataloader:
            x_t = x_t.to(device)
            y_t = y_t.to(device)

            output_t = model(x_t)
            l_t = loss(output_t, y_t)
            error += l_t.mean().item() * x_t.size(0)
            num_test += x_t.size(0)
        print('epoch %d, loss %f' % (i + 1, error / num_test))

# visualization
for x_t, y_t in test_dataloader:
    x_cu = x_t.to(device)

    output_t = model(x_cu)
    _, preds = torch.max(output_t, 1)
    x_t = x_t.numpy()

    fig = plt.figure(figsize=(25,4))
    for i in range(batch_size):
        ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_t[i]), cmap='gray')
        ax.set_title("{} ({})".format(str(preds[i].item()), str(y_t[i].item())),
                     color=("green" if preds[i] == y_t[i] else "red"))
    plt.show()
