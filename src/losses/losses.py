import torch.nn as nn
import torch

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        return loss


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pred, target):
        loss = torch.sqrt(self.mse(pred, target) + self.eps)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        loss = self.l1(pred, target)
        return loss
