from torch.nn import functional as F
import torch
import torch.nn as nn
from pytorch_metric_learning import losses

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))




def get_loss(loss_name: str):
    if loss_name == 'crossentropy':
        return F.cross_entropy
    elif loss_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif loss_name == 'Focal':
        return FocalLoss(gamma=2)
    elif loss_name == 'logBCE':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'bce':
        return nn.BCELoss()
    elif loss_name == 'CL':
        return SupervisedContrastiveLoss()
    elif loss_name == 'TP':
        return nn.TripletMarginLoss(margin=1.0, p=2)
    else:
        print(f'{loss_name}: invalid loss name')
        return
