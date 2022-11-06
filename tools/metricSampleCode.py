from pytorch_metric_learning import losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ### 
from torchvision import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import ipdb

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ### 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ### 
def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        #ipdb.set_trace()
        #embeddings = torch.nonzero(embeddings,as_tuple=False)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 40 == 0:
            print("Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(epoch, batch_idx, loss, mining_func.num_triplets))

### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(dataset, model, accuracy_calculator):
    embeddings, labels = get_all_embeddings(dataset, model)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(embeddings, 
                                                embeddings,
                                                np.squeeze(labels),
                                                np.squeeze(labels),
                                                True)
    print("Test set accuracy (MAP@10) = {}".format(accuracies["mean_average_precision_at_r"]))

device = torch.device("cuda")

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

batch_size = 200

dataset1 = datasets.MNIST('.', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('.', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1


### pytorch-metric-learning stuff ###
#distance = distances.CosineSimilarity()
#reducer = reducers.ThresholdReducer(low = 0)
#loss_func = losses.TripletMarginLoss(margin = 0.2, distance = distance, reducer = reducer)
loss_func = losses.TripletMarginLoss(margin = 0.2)
#mining_func = miners.TripletMarginMiner(margin = 0.2, distance = distance, type_of_triplets = "semihard")
mining_func = miners.TripletMarginMiner(margin = 0.2, type_of_triplets = "semihard")
accuracy_calculator = AccuracyCalculator(include = ("mean_average_precision_at_r",), k = 10)
### pytorch-metric-learning stuff ###


for epoch in range(1, num_epochs+1):
    train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
    test(dataset2, model, accuracy_calculator)

