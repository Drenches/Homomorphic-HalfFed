import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tenseal as ts
from utils import *


class EncConvNet:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()
        
        
    def forward(self, enc_x, windows_nb):
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        # fc1 layer
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        # square activation
        enc_x.square_()
        # fc2 layer
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SimpleConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(SimpleConvNet, self).__init__()        
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        # the model uses the square activation function
        x = x * x
        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x

class Encrypt(torch.nn.Module):
    def __init__(self, torch_nn):
        super(Encrypt, self).__init__()        
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
  
    def forward(self, enc_x, windows_nb):
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        return enc_x
  
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ServerNetMNIST(torch.nn.Module):
    def __init__(self):
        super(ServerNetMNIST, self).__init__()        
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
  
    def forward(self, x):
        x = self.conv1(x)
        x = x * x
        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        return x
  
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class UserNetMNIST(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(UserNetMNIST, self).__init__()        
        self.fc1 = torch.nn.Linear(256, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)
  
    def forward(self, x):
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x
  
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ServerNetCIFAR10(nn.Module):
    def __init__(self):
        super(ServerNetCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False) #1
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.Sequential(
            ResNetBlock(16, 16, stride=1), #3
            ResNetBlock(16, 16, stride=1) #5
        )
        self.layer2 = nn.Sequential(
            ResNetBlock(16, 32, stride=2), #7
            ResNetBlock(32, 32, stride=1) #9
        )
        self.layer3 = nn.Sequential(
            ResNetBlock(32, 64, stride=2), #11
            ResNetBlock(64, 64, stride=1) #13
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        return out


class UserNetCIFAR10(nn.Module):
    def __init__(self, output=10):
        super(UserNetCIFAR10, self).__init__()
        self.layer4 = nn.Sequential(
            ResNetBlock(16, 16, stride=1), #15
            ResNetBlock(16, 16, stride=1), #17
            ResNetBlock(16, 16, stride=1) #19
        )
        self.layer5 = nn.Sequential(
            ResNetBlock(16, 32, stride=2), #21
            ResNetBlock(32, 32, stride=1), #23
            ResNetBlock(32, 32, stride=1) #25
        )
        self.layer6 = nn.Sequential(
            ResNetBlock(32, 64, stride=2), #27
            ResNetBlock(64, 64, stride=1), #29
            ResNetBlock(64, 64, stride=1) #31
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, output) #32

    def forward(self, x):
        out = self.layer4(x)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# class ResNetBlock(nn.Module):
# # this implementation of ResBlock considers split a resblock into two parties(server and client)
#     def __init__(self, in_channels, out_channels, stride=1, server_part=False):
#         super(ResNetBlock, self).__init__()
#         self.server_part = server_part
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         if self.server_part:
#             out = self.conv1(x)
#             out = self.bn1(out)
#             out = F.relu(out)
#             out = self.conv2(out)
#             out = self.bn2(out)
#         else:
#             out = F.relu(x)
#             out = self.conv1(out)
#             out = self.bn1(out)
#             out = F.relu(out)
#             out = self.conv2(out)
#             out = self.bn2(out)
#         shortcut = self.shortcut(x)
#         out += shortcut
#         out = F.relu(out)
#         return out