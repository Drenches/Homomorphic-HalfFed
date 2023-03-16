import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import tenseal as ts
from utils import *
import pdb
import datetime
## Load data and model
gpu = torch.cuda.is_available()
gpu = False
entire_model = ConvNet()

user_model = UserNet(entire_model)
server_model = ServerNet(entire_model)
batch_size = 1
num_clients = 10
train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

server_model.cuda()
user_model.cuda()

# Split the dataset into multiple clients
client_datasets = torch.utils.data.random_split(train_data, [len(train_data)//num_clients]*num_clients)

# Define the loss function and the server's optimizer
criterion = torch.nn.CrossEntropyLoss()
server_optimizer = torch.optim.Adam(user_model.parameters(), lr=0.01)

# Train the local models
for epoch in range(10):
    # Iterate over each client's dataset
    for client_id, client_dataset in enumerate(client_datasets):
        # Create a dataloader for the client's dataset
        client_dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        
        # Train the user model on the client's dataset
        user_optimizer = torch.optim.Adam(user_model.parameters(), lr=0.001)
        for images, labels in client_dataloader:
            images, labels = images.cuda(), labels.cuda()
            
            # Forward pass through the server model
            front_output = server_model(images)
            
            # Forward pass through the user model
            user_output = user_model(front_output).view(1, -1)
            
            # Calculate the loss and perform backpropagation
            loss = criterion(user_output, labels)
            user_optimizer.zero_grad()
            loss.backward()
            user_optimizer.step()
        
        # Send the user model's gradient to the server
        user_gradient = {param_name: param.grad for param_name, param in user_model.named_parameters()}
        torch.distributed.rpc.rpc_sync("client_{}".format(client_id), server_model.update_gradient, args=(user_gradient,))

        # Aggregate the gradients and update the server model
        server_gradient = server_model.aggregate_gradients()
        server_optimizer.zero_grad()
        for param_name, param in server_model.front_part.named_parameters():
            if param_name in server_gradient:
                param.grad = server_gradient[param_name]
        server_optimizer.step()



    