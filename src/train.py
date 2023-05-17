import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

from pytorch_metric_learning import losses, miners, distances, reducers

from model import TakeoffClassModel
from dataset import ImageDataset

df = pd.read_csv('image_data.csv')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = ImageDataset(df, transform=transform)

n_train_ratio = 80

n_train = int(len(dataset) * n_train_ratio / 100)
n_val = int(len(dataset) - n_train)

train, val = torch.utils.data.random_split(dataset, [n_train, n_val])

batch_size = 16

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

def train(model, loss_func, mining_func, device, dataloader, optimizer, epoch):
    model.train() 
    for idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(inputs)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print('Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}'.format(epoch, idx, loss, mining_func.num_triplets))
    print()

def test(model, dataloader, device):
    _predicted_metrics = []
    _true_labels = []
    model.eval()
    with torch.no_grad():    
        for i, (inputs,  labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            metric = model(inputs).detach().cpu().numpy()
            metric = metric.reshape(metric.shape[0], metric.shape[1])
            _predicted_metrics.append(metric)
            _true_labels.append(labels.detach().cpu().numpy())
    return np.concatenate(_predicted_metrics), np.concatenate(_true_labels)

epochs = 50
laerning_rate = 1e-4
print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TakeoffClassModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1)
test_predicted_metrics = []
test_true_labels = []
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")

for epoch in range(1, epochs + 1):
    print('Epoch {}/{}'.format(epoch, epochs))
    print('-' * 10)
    train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
    _tmp_metrics, _tmp_labels = test(model, val_loader, device)
    test_predicted_metrics.append(_tmp_metrics)
    test_true_labels.append(_tmp_labels)
print('Finished Training')

# visualize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#perplexityはtest_dataが30以上あるときは指定しなくてよい
tSNE_metrics = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(test_predicted_metrics[-1])

plt.scatter(tSNE_metrics[:, 0], tSNE_metrics[:, 1], c=test_true_labels[-1])
plt.colorbar()
plt.savefig("output.png")
plt.show()

torch.save(model.state_dict(), 'model.pth')