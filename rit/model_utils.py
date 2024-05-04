import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from config import DEVICE



class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d,self).__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs,dim=1), targets)




class GeneralizedDiceLoss(nn.Module):
    # Author: Rakshit Kothari
    # Input: (B, C, ...)
    # Target: (B, C, ...)
    def __init__(self, epsilon=1e-5, weight=None, softmax=True, reduction=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.weight = []
        self.reduction = reduction
        if softmax:
            self.norm = nn.Softmax(dim=1)
        else:
            self.norm = nn.Sigmoid()

    def forward(self, ip, target):

        # Rapid way to convert to one-hot. For future version, use functional
        Label = (np.arange(4) == target.cpu().numpy()[..., None]).astype(np.uint8)
        target = torch.from_numpy(np.rollaxis(Label, 3,start=1)).cuda()

        assert ip.shape == target.shape
        ip = self.norm(ip)

        # Flatten for multidimensional data
        ip = torch.flatten(ip, start_dim=2, end_dim=-1).cuda().to(torch.float32)
        target = torch.flatten(target, start_dim=2, end_dim=-1).cuda().to(torch.float32)
        
        numerator = ip*target
        denominator = ip + target

        class_weights = 1./(torch.sum(target, dim=2)**2).clamp(min=self.epsilon)

        A = class_weights*torch.sum(numerator, dim=2)
        B = class_weights*torch.sum(denominator, dim=2)

        dice_metric = 2.*torch.sum(A, dim=1)/torch.sum(B, dim=1)
        if self.reduction:
            return torch.mean(1. - dice_metric.clamp(min=self.epsilon))
        else:
            return 1. - dice_metric.clamp(min=self.epsilon)