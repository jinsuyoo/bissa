from .losses import MSELoss, RMSELoss, L1Loss
from torch import nn
from typing import Union


def create_loss(loss_name):
    # type: (str) -> Union[nn.Module, None]
    loss_cls = getattr(losses, loss_name, None)
    if loss_cls is None:
        return None
    return loss_cls()
