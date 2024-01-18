import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import GlobalData
from Backtest import Factor_backtest
import torch
import torch.nn as nn


features = torch.load('./features.pt').float()
rr = torch.load('./return_rate.pt').float()
rr = rr.view(-1, 1)
n_features = features.shape[1]

class MyModel(nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(in_features = input_dim, out_features = 20)
        self.fc2 = nn.Linear(in_features = 20, out_features = 30)
        self.fc3 = nn.Linear(in_features = 30, out_features = 1)
        #默认带梯度

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

nepoch = 10
batch_size = 200
lr = 0.001

np.random.seed(123)
torch.manual_seed(123)

model = MyModel(n_features)
losses_sgd = []

# 完成模型训练
opt = torch.optim.SGD(model.parameters(), lr=lr)

n = features.shape[0]
obs_id = np.arange(n)  # [0, 1, ..., n-1]

mean = features.mean(dim=0)
std = features.std(dim=0)
features = (features - mean)/std

# Run the whole data set `nepoch` times
for i in range(nepoch):
    # Shuffle observation IDs
    np.random.shuffle(obs_id)

    # Update on mini-batches
    for j in range(0, n, batch_size):
        # Create mini-batch
        x_mini_batch = features[obs_id[j:(j + batch_size)]]
        y_mini_batch = rr[obs_id[j:(j + batch_size)]]
        # Compute loss
        pred = model(x_mini_batch)
        loss = nn.MSELoss()(pred, y_mini_batch)
        # Compute gradient and update parameters
        opt.zero_grad()
        loss.backward()
        opt.step()  # update the parameters of the optimizer
        losses_sgd.append(loss.item())

        if (j // batch_size) % 200 == 0:
            print(f"epoch {i}, batch {j // batch_size}, loss = {loss.item()}")

plt.plot(losses_sgd)