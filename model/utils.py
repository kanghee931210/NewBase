import ml_fastvit.models
from timm.models import create_model
import torch
import torch.nn as nn
from ml_fastvit.models.modules.mobileone import reparameterize_model


def get_model(model_name:str):
    if model_name == 'resnet18':
        model = create_model('resnet18',pretrained=True)
        return model
    elif model_name == 'resnet34':
        model = create_model('resnet34', pretrained=True)
        return torch.nn.Sequential(*(list(model.children())[:-1]))
    elif model_name == 'resnet50':
        model = create_model('resnet50', pretrained=True)
        return torch.nn.Sequential(*(list(model.children())[:-1]))
    elif model_name == 'b0':
        model = create_model('efficientnet_b0', pretrained=True)
        return model
    elif model_name == 'f-vit-sa-24':
        model = create_model('fastvit_sa24')
        return model
    else:
        raise ValueError(f'Model name {model_name} is not valid.')

if __name__ == '__main__':
    pass