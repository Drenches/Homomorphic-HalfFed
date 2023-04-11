import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from utils import *
from nets import *
import pdb
import datetime
from fedlab.utils.dataset import MNISTPartitioner,CIFAR10Partitioner
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import WeightedRandomSampler

## Load data and model
gpu = torch.cuda.is_available()
# gpu = False
# entire_model = SimpleConvNet()
batch_size = 512
major_classes_num=3
num_clients = 10
seed = 2021
# data_name = 'MNIST'
data_name = 'CIFAR10'
if data_name=='MNIST':
    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    # Split the dataset into multiple clients
    # This part is implenmented by FedLab: https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html
    # major class division
    # client_datasets_part = MNISTPartitioner(
    #     train_data.targets,
    #     num_clients=num_clients,
    #     partition="noniid-#label",
    #     major_classes_num=major_classes_num,
    #     seed = seed
    # )

    # dir alpha class division
    client_datasets_part = MNISTPartitioner(train_data.targets, 
                                        num_clients=num_clients,
                                        partition="noniid-labeldir", 
                                        dir_alpha=0.5,
                                        seed=seed)
    
    #load server model
    server_model_init = ServerNetMNIST().cuda()
    server_model_list = [copy.deepcopy(server_model_init) for _ in range(num_clients)]

if data_name== 'CIFAR10':
    train_data = datasets.CIFAR10('data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.CIFAR10('data', train=False, download=True, transform=transforms.ToTensor())
    # Split the dataset into multiple clients
    
    #iid
    client_datasets_part  = CIFAR10Partitioner(train_data.targets,
                                      num_clients,
                                      balance=True,
                                      partition="iid",
                                      seed=seed)

    # shard division
    # num_shards = 200
    # client_datasets_part = CIFAR10Partitioner(
    #     train_data.targets,
    #     num_clients=num_clients,
    #     balance=None,
    #     partition="shards",
    #     num_shards=num_shards,
    #     seed = seed
    # )

    # dir alpha class division
    # client_datasets_part = CIFAR10Partitioner(train_data.targets, 
    #                                     num_clients=num_clients,
    #                                     balance=None,
    #                                     partition="dirichlet", 
    #                                     dir_alpha=0.5,
    #                                     seed=seed)
    
    # load server model
    server_model_init = ServerNetCIFAR10().cuda()
    server_model_list = [copy.deepcopy(server_model_init) for _ in range(num_clients)]

num_classes = torch.bincount(torch.Tensor(test_data.targets).int()).shape[0]

# User model list
if data_name=='MNIST':
    client_model_list = [UserNetMNIST(output=num_classes).cuda().train() for _ in range(num_clients)]
if data_name=='CIFAR10':
    client_model_list = [UserNetCIFAR10(output=num_classes).cuda().train() for _ in range(num_clients)]

# client_datasets = torch.utils.data.random_split(train_data, [len(train_data)//num_clients]*num_clients)
client_dataloaders = [torch.utils.data.DataLoader(train_data, sampler=SubsetRandomSampler(client_datasets_part[i]), batch_size=batch_size) for i in range(num_clients)]

# Define the loss function and the server's optimizer
criterion = torch.nn.CrossEntropyLoss()
server_optimizer_list = [torch.optim.Adam(server_model.parameters(), lr=0.001) for server_model in server_model_list]
client_optimizer_list = [torch.optim.Adam(client_model.parameters(), lr=0.001) for client_model in client_model_list]

def open_train_mode(model_list):
    for model in model_list:
        model.train()
# Muti-round training
def train(round = 200, epoches = 100,  server_model_list= server_model_list):
    server_stop_marker = 0
    distribution_list = [torch.zeros(num_classes).cuda() for _ in range(num_clients)] # record labels distribution
    for i in range(round):
        print('round:', i)
        for client_id in range(num_clients):

            # Train multiple epoches for each client
            acc_counter = 0

            # Load server model corresonding for each specific client
            server_model = server_model_list[client_id].train()
            server_optimizer =server_optimizer_list[client_id]

            # Load client's dataset
            client_dataloader = client_dataloaders[client_id]
            client_optimizer = client_optimizer_list[client_id]
            client_model = client_model_list[client_id].train()
            
            num_correct = 0
            for epoch in range(epoches):

                client_optimizer.zero_grad()
                server_optimizer.zero_grad()

                # Train the user model on the client's dataset
                images, labels = next(iter(client_dataloader))
                images, labels = images.cuda(), labels.cuda()
                distribution_list[client_id] += torch.bincount(labels.flatten(), minlength=num_classes)

                # Forward pass through the server model
                front_output = server_model(images)
                
                # Forward pass through the user model
                user_output = client_model(front_output)
                loss = criterion(user_output, labels)

                # Calculate the loss and perform backpropagation

                loss.backward()

                client_optimizer.step()
                server_optimizer.step()
                
                _, pred = torch.max(user_output, 1)
                correct = np.squeeze(pred.eq(labels.data.view_as(pred)))

            # Calculate the train accuracy of each client
            print('Client ', client_id)
            acc_counter += train_acc(user_output, labels)
        

        # Aggregate and update the server model
        # server_model = aggregation(server_model_list)
        # server_model_list = [copy.deepcopy(server_model) for _ in range(num_clients)]
        # print('server model aggregated')
        
        test_acc = test(distribution_list=distribution_list.copy(), server_model_list= server_model_list)
        # if i%10==0:  
            
        
        # if i%10==0:  
        #     test_acc = test(distribution_list=distribution_list.copy())
        #     if abs(test_acc-acc_counter/num_clients)>0.2:
        #         server_stop_marker = 1
        # if server_stop_marker: 
        #     server_model.eval()
        #     print('Server part stops updating')
        # else:
        #     server_model.train()
        #     server_optimizer.step()
        #     print('Server part updated')
        
        # # Alternate update
        # if i%10==0:  
        #     test_acc = test(distribution_list=distribution_list.copy())
        #     if abs(test_acc-acc_counter/num_clients)>0.2:
        #         server_stop_marker = 1
        # if server_stop_marker: 
        #     server_model.eval()
        #     print('Server part stops updating')
        # else:
        #     server_optimizer.step()
        #     print('Server part updated')
        
        

def test(test_batch_size = 2048, distribution_list=None, server_model_list=None):
    distribution = TestSampGen(test_data, distribution_list)
    total_correct = 0
    total_vol = 0
    for client_id in range(num_clients):
        client_model = client_model_list[client_id].eval()
        server_model = server_model_list[client_id].eval()
        # test_loader = torch.utils.data.DataLoader(test_data, sampler=SubsetRandomSampler(test_part[client_id]), batch_size=test_batch_size)
        sampler = WeightedRandomSampler(weights=distribution[client_id].tolist(), replacement=True, num_samples=len(test_data)//num_clients)
        test_loader = torch.utils.data.DataLoader(test_data, sampler=sampler, batch_size=test_batch_size)
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
            pdb.set_trace()
            for i in range(len(labels)):
                label = labels.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            total_vol += labels.shape[0]

        total_correct += torch.sum(correct).item()
        test_loss = test_loss/labels.shape[0]
        print('Client ', client_id)
        print(f'Test Loss: {test_loss:.6f}')
        print(
            f'Test Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% ' 
            f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})\n'
        )
    return total_correct / total_vol

train(round = 1000, epoches = 10, server_model_list=server_model_list)
# Final test
print('\n ### Final Test ###')
# test_part = MNISTPartitioner(
#     test_data.targets,
#     num_clients=num_clients,
#     partition="noniid-#label",
#     major_classes_num=major_classes_num,
#     seed = seed
# )
pdb.set_trace()

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

    