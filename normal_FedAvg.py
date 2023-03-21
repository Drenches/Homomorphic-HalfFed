import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import tenseal as ts
from utils import *
from fedlab.utils.dataset import MNISTPartitioner
from torch.utils.data import SubsetRandomSampler
import copy
import pdb
import datetime

## Load data and model
gpu = torch.cuda.is_available()
model = ConvNet().cuda()

batch_size = 512
num_clients = 10
num_epochs = 500
major_classes_num = 9
train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

# Split the dataset into multiple clients
# client_datasets_part = MNISTPartitioner(
#     train_data.targets,
#     num_clients=num_clients,
#     partition="noniid-#label",
#     major_classes_num=major_classes_num
# )
client_datasets_part = MNISTPartitioner(
    train_data.targets,
    num_clients=num_clients,
    partition="iid",
)
# test_part = MNISTPartitioner(
#     test_data.targets,
#     num_clients=num_clients,
#     partition="noniid-#label",
#     major_classes_num=major_classes_num
# )
test_part = MNISTPartitioner(
    test_data.targets,
    num_clients=num_clients,
    partition="iid",
)

# Define FL process 
criterion = torch.nn.CrossEntropyLoss()
def train(model, train_loader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_acc(output, target)
    return model

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    counter = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += torch.sum(criterion(output, target)).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            counter += 1
    test_loss /= counter
    accuracy = correct / len(test_loader.dataset)
    pdb.set_trace()
    return test_loss, accuracy

def federated_train(model, num_epochs=5, num_clients=10):
    # global_weights = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        local_models = []
        for client_id in range(num_clients):
            # Evaluate global model on local test data every 10 epoches
            if epoch%10  == 0:
                test_loader = torch.utils.data.DataLoader(train_data, sampler=SubsetRandomSampler(client_datasets_part[client_id]), batch_size=2048)
                test_loss, accuracy = test(model, test_loader)
                print(f'Client {client_id + 1}, Test loss: {test_loss:.4f}, Accuracy: {accuracy*100:.5f}%')

            # Get local model and train on client data
            local_model = copy.deepcopy(model)
            train_loader = torch.utils.data.DataLoader(train_data, sampler=SubsetRandomSampler(client_datasets_part[client_id]), batch_size=batch_size)
            local_model = train(local_model, train_loader)
            local_models.append(local_model)

        # Average the local models to get the updated global model weights
        global_weights = {}
        for k in local_models[0].state_dict().keys():
            global_weights[k] = torch.stack([local_models[i].state_dict()[k].float() for i in range(len(local_models))]).mean(0)

        # Update the global model with the new weights
        model.load_state_dict(global_weights)

        
federated_train(model, num_epochs=num_epochs, num_clients=num_clients)
        


