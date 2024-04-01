import os, sys

sys.path.append(os.path.join(os.getcwd(), "ssa/modules/stn"))
sys.path.append(os.path.join(os.getcwd(), "ssa/modules/car"))
sys.path.append(os.path.join(os.getcwd(), "ssa/modules/vdsr"))
sys.path.append(os.path.join(os.getcwd(), "ssa/modules/sr_models/models"))

import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

import utils
from utils import imresize
from EDSR.edsr import EDSR
from modules import DSN
from stn_model import SpatialTransformerNet

# from edsr_model import Net as EDSR
# from rcan_model import Net as RCAN
from adaptive_gridsampler.gridsampler import Downsampler
from skimage.color import rgb2ycbcr


class MaskingNetwork(nn.Module):
    def __init__(self, opt):
        super(MaskingNetwork, self).__init__()
        self.vdsr = torch.load("ssa/modules/vdsr/model/mask/model_epoch_30.pth")[
            "model"
        ]
        self.n_seq = opt["n_seq"]

    # @torch.no_grad()
    def forward(self, x):
        b, c, h, w = x.size()  # (4, 3, 256, 256)
        mask = self.vdsr(x)
        return mask
