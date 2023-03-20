import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import tenseal as ts
from utils import *
import pdb
import datetime
from fedlab.utils.dataset import MNISTPartitioner
from torch.utils.data import SubsetRandomSampler


## Load data and model
gpu = torch.cuda.is_available()
# gpu = False
entire_model = ConvNet()

batch_size = 512
num_clients = 10
major_classes_num=1
train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())


server_model = ServerNet().cuda()
# Split the dataset into multiple clients
client_datasets_part = MNISTPartitioner(
    train_data.targets,
    num_clients=num_clients,
    partition="noniid-#label",
    major_classes_num=major_classes_num
)
# client_datasets = torch.utils.data.random_split(train_data, [len(train_data)//num_clients]*num_clients)

# User model list
client_model_list = [UserNet().cuda().train() for _ in range(num_clients)]

# Define the loss function and the server's optimizer
criterion = torch.nn.CrossEntropyLoss()
server_optimizer = torch.optim.Adam(server_model.parameters(), lr=0.01)
client_optimizer_list = [torch.optim.Adam(client_model.parameters(), lr=0.001) for client_model in client_model_list]


# Muti-round training
server_model.train()
round = 150
for i in range(round):
    print('round:', i)

    # One batch iteration across each client
    server_optimizer.zero_grad()
    for client_id in range(num_clients):
    # for client_id, client_dataset in enumerate(client_datasets):
        # Create a dataloader for the client's dataset
        client_dataloader = torch.utils.data.DataLoader(train_data, sampler=SubsetRandomSampler(client_datasets_part[client_id]), batch_size=batch_size)
        client_optimizer = client_optimizer_list[client_id]
        client_model = client_model_list[client_id]
        # Train the user model on the client's dataset
        images, labels = next(iter(client_dataloader))
        images, labels = images.cuda(), labels.cuda()
        
        # Forward pass through the server model
        front_output = server_model(images)
        
        # Forward pass through the user model
        user_output = client_model(front_output)

        # Calculate the loss and perform backpropagation
        loss = criterion(user_output, labels)
        client_optimizer.zero_grad()
        loss.backward()
        client_optimizer.step()

        # Calculate the train and test accuracy
        print('Client ', client_id)
        train_acc(user_output, labels)

    # Aggregate the gradients and update the server model
    server_optimizer.step()


# Final test
print('\n ### Final Test ###')
test_part = MNISTPartitioner(
    test_data.targets,
    num_clients=num_clients,
    partition="noniid-#label",
    major_classes_num=major_classes_num
)
test_batch_size = 1024
server_model.eval()
for client_id in range(num_clients):
    client_model = client_model_list[client_id].eval()
    test_loader = torch.utils.data.DataLoader(train_data, sampler=SubsetRandomSampler(client_datasets_part[client_id]), batch_size=batch_size)
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        server_output = server_model(images)
        user_output = client_model(server_output)
        loss = criterion(user_output, labels)
        test_loss += loss.item()
        _, pred = torch.max(user_output, 1)
        correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
        for i in range(len(labels)):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    test_loss = test_loss/len(test_loader)
    print('Client ', client_id)
    print(f'Test Loss: {test_loss:.6f}')
    print(
        f'Test Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% ' 
        f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})\n'
    )



# Send the user model's gradient to the server
# user_gradient = {param_name: param.grad for param_name, param in user_model.named_parameters()}
# torch.distributed.rpc.rpc_sync("client_{}".format(client_id), server_model.update_gradient, args=(user_gradient,))

# # Aggregate the gradients and update the server model
# server_gradient = server_model.aggregate_gradients()
# server_optimizer.zero_grad()
# for param_name, param in server_model.front_part.named_parameters():
#     if param_name in server_gradient:
#         param.grad = server_gradient[param_name]
# server_optimizer.step()

    