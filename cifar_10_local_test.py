import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from utils import *
from nets import *
from cifar10_data_loader import *
import pdb
from torch.utils.data import DataLoader
import datetime
from fedlab.utils.dataset import MNISTPartitioner,CIFAR10Partitioner
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import WeightedRandomSampler
import random

## Load data and model
gpu = torch.cuda.is_available()
total_num_classes = 10
batch_size = 128
num_clients = 100
# seed = 2021
data_name = 'CIFAR10'

selected_data = True
# selected_data = False
if selected_data:
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, DATA_CLASS = load_partition_data_cifar10('data', num_clients, 10, batch_size)
    client_id = 8
    client_dataloader = train_data_local_dict[client_id]
else:
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    transform_train.transforms.append(Cutout(16))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


    train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)

    test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024,
                                            shuffle=True)

learningRate = 1e-3
eps = 1e-3
AMSGrad = True
# User model list
# client_model = CIFAR10CNNUser().cuda().train()
client_model = SimpleUserNet().cuda().train()
# client_model = UserNetCIFAR10().cuda().train()
client_optimizer =torch.optim.Adam(params = client_model.parameters(), lr = learningRate, eps = eps, amsgrad = AMSGrad)

# Server model initi
# server_model = CIFAR10CNNServer().cuda().train()
server_model = SimpleServerNet().cuda().train()
# server_model = ServerNetCIFAR10().cuda().train()
server_optimizer = torch.optim.Adam(params = server_model.parameters(), lr = learningRate, eps = eps, amsgrad = AMSGrad)

criterion = nn.CrossEntropyLoss()

# training
def train(epoches = 100, p=0.1, server_model = server_model, client_model = client_model):
    
    for epoch in range(epoches):
        correct = 0
        total = 0
        train_loss = 0
        avg_train_acc = []

        if selected_data:
            client_dataloader = train_data_local_dict[client_id]
        else:
            client_dataloader = train_loader

        # Train the user model on the client's dataset
        for _, (images, labels) in enumerate(client_dataloader):
            
            if selected_data:
                images, labels = images.permute(0, 3, 1, 2).cuda(), labels.cuda()
            else:
                images, labels = images.cuda(), labels.cuda()
        
            
            # Forward pass through the server model
            front_output = server_model(images)
            
            # Forward pass through the user model
            user_output = client_model(front_output)

            # Calculate the loss and perform backpropagation
            client_optimizer.zero_grad()
            server_optimizer.zero_grad()

            loss = criterion(user_output, labels)
            loss.backward()

            client_optimizer.step()
            server_optimizer.step()

            _, pred = torch.max(user_output, 1)
            correct += torch.sum(np.squeeze(pred.eq(labels.data.view_as(pred))))
            train_loss += loss.item()
            total += labels.shape[0]

            # print(torch.sum(np.squeeze(pred.eq(labels.data.view_as(pred))))/labels.shape[0])
        
        # pdb.set_trace()
        # Calculate the train accuracy of each client
        print('\n### Train Loss: %.4f,  Train Acc.: %.4f ###' % (train_loss/total , correct/total))
        avg_train_acc.append(correct/total)
    

        with torch.no_grad():
            test_acc = []
            correct = 0
            total = 0
            test_loss = 0
            server_model = server_model.eval()
            client_model = client_model.eval()
            if selected_data:
                client_test_loader = test_data_local_dict[client_id]
            else:
                client_test_loader =  torch.utils.data.DataLoader(test_data, batch_size=1024,
                                            shuffle=True)
            for images, labels in client_test_loader:
                if selected_data:
                    images, labels = images.permute(0, 3, 1, 2).cuda(), labels.cuda()
                else:
                    images, labels = images.cuda(), labels.cuda()
                front_ouput = server_model(images)
                output = client_model(front_ouput)
                loss = criterion(output, labels)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss += loss.item()

            server_model.train()
            client_model.train()
            print('### Avg. Test Acc. : %4f ###' % ( correct/total ))
        
train()
