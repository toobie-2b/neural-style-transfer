import torch
import torch.nn as nn

def get_features(x, model, layers):
    '''
    Returns a dictionary with features mapped to the corresponding
    layer names.\n
    `x`: Tensor for which the features need to be generated.\n
    `model`: The model which is supposed to calculate the features.
    '''

    features = {}
    for name, layer in model._modules.items():
        x = layer(x)

        if name in layers:
            features[layers[name]] = x

    return features

def gram_matrix(tensor):
    '''
    Returns the gram matrix of the input tensor
    '''

    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram