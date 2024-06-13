import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler
import tenseal as ts
import copy
import random
import pdb

def ServerInference(enc_model, x_enc, windows_nb, kernel_shape, stride):
    # Encrypted evaluation
    enc_output = enc_model(x_enc, windows_nb)
    return enc_output

def train_acc(output, target):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    
    # calculate train accuracy for each object class
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
    
    print(
        f'Train Accuracy (Overall): {np.sum(class_correct) / np.sum(class_total)} '
        )
    # f'Train Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% \n' 
    # f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
    return np.sum(class_correct) / np.sum(class_total)

def GenCiph(data, context, kernel_shape, stride):
    x_enc, windows_nb = ts.im2col_encoding(
            context, data.view(28, 28).tolist(), kernel_shape[0],
            kernel_shape[1], stride
        )
    return x_enc, windows_nb

def TestSampGen(data, distribution):
    class_counts = torch.bincount(torch.Tensor(data.targets).int()).cuda()
    class_weights = 1.0 / class_counts
    for i in range(len(distribution)):
        skew_weights =  distribution[i]*class_weights
        sample_weights = skew_weights[data.targets]
        distribution[i] = sample_weights
    return distribution

# def aggregation(client_models):
#     average_param = client_models[0].parameters()
#     # for i in range(0, 1):
#     for i in range(0, len(client_models)):
#         for k in average_param.keys():
#             if i ==0:
#                 continue
#             else:
#                 average_param[k] = client_models[]
#             for global_param, client_param in zip(global_model.parameters(), client_models[i].parameters()):
#                 global_param.data += client_param.data

#     for global_param in global_model.parameters():
#         global_param.data /= len(client_models)
#     return client_models[0]


def aggregation(model_list):

    # 创建一个新的模型
    new_model = copy.deepcopy(model_list[0])
    new_model.load_state_dict(model_list[0].state_dict())

    # 获取所有模型的参数
    params_list = [model.state_dict() for model in model_list]

    # 对于每个参数，计算平均值并设置为新模型的参数
    for param_name in new_model.state_dict():
        param_sum = None
        param_count = 0

        # 对于每个模型，将该参数添加到总和中并增加计数器
        for model_params in params_list:
            if param_name in model_params:
                if param_sum is None:
                    param_sum = model_params[param_name]
                else:
                    param_sum += model_params[param_name]
                param_count += 1

        # 如果存在该参数，则将总和除以计数器，得到平均值，并设置为新模型的参数
        if param_sum is not None:
            new_param = param_sum / param_count
            new_model.state_dict()[param_name].copy_(new_param)

    return new_model

class GroupSampler(Sampler):
    def __init__(self, data_source, group_size):
        self.data_source = data_source
        self.group_size = group_size
        self.indices = list(range(len(self.data_source)))
    
    def __iter__(self):
        # random.shuffle(self.indices)
        grouped_indices = [self.indices[i:i + self.group_size] for i in range(0, len(self.indices), self.group_size)]
        random.shuffle(grouped_indices)
        flattened_indices = [idx for group in grouped_indices for idx in group]
        return iter(flattened_indices)
    
    def __len__(self):
        return len(self.data_source)

class GroupBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch