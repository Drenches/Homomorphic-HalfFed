import torch
import numpy as np
import tenseal as ts
from PIL import Image
import copy
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
        f'Train Accuracy (Overall): {np.sum(class_correct) / np.sum(class_total)} \n'
        # f'Train Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% \n' 
        # f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
        )
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

def aggregation(client_models):
    global_model = copy.deepcopy(client_models[0])
    for i in range(1, len(client_models)):
        for global_param, client_param in zip(global_model.parameters(), client_models[i].parameters()):
            global_param.data += client_param.data

    for global_param in global_model.parameters():
        global_param.data /= len(client_models)
    return global_model


def save_image(image_tensor, filename='/home/dev/workspace/Homomorphic-HalfFed/checkimage.png'):

    image_np = np.uint8(image_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255)

    image_pil = Image.fromarray(image_np)

    if filename.lower().endswith('.png'):
        image_pil.save(filename)
    elif filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
        image_pil.save(filename, quality=95)
    else:
        raise ValueError("Unsupported file format: %s" % filename)