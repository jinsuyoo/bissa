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


class PreDownscaleNetwork(nn.Module):
    def __init__(self, opt):
        super(PreDownscaleNetwork, self).__init__()
        self.stn = SpatialTransformerNet()
        # self.vdsr = torch.load("ssa/modules/vdsr/model/rgb/patch_epoch_23.pth")["model"]
        # self.edsr = torch.load("ssa/modules/sr_models/checkpoints/edsr/model_epoch_30.pth")["model"]
        # self.rcan = torch.load("ssa/modules/sr_models/checkpoints/rcan/model_epoch_30.pth")["model"]
        self.n_seq = opt["n_seq"]

    # @torch.no_grad()
    def forward(self, x):
        b, _, c, h, w = x.size()  # (4, 7, 3, 256, 256)
        hr = torch.stack(
            [p for p in x]
            # [self.stn(p) for p in x]
            # [p + self.stn(p) for p in x]
            # [self.vdsr(p) for p in x]
            # [self.edsr(p) for p in x]
            # [self.rcan(p) for p in x]
        )
        # print("\nx  : ", torch.mean(x), torch.std(x), flush = True)
        # print("hr : ", torch.mean(hr), torch.std(hr), flush = True)
        lr = torch.stack([imresize(p, scale=0.25) for p in hr])
        return lr, hr[:, self.n_seq // 2, :, :, :]


class PostDownscaleNetwork:
    def __init__(self):
        self.SCALE = 4
        self.KSIZE = 3 * self.SCALE + 1
        self.OFFSET_UNIT = self.SCALE
        self.model_dir = "./ssa/modules/car/models"
        self.img_dir = "./ssa/modules/car/images"
        self.output_dir = "./ssa/modules/car/results"

        self.init_network()

    def init_network(self):
        kernel_generation_net = DSN(k_size=self.KSIZE, scale=self.SCALE).cuda()
        downsampler_net = Downsampler(self.SCALE, self.KSIZE).cuda()
        upscale_net = EDSR(32, 256, scale=self.SCALE).cuda()

        kernel_generation_net = nn.DataParallel(kernel_generation_net, [0])
        downsampler_net = nn.DataParallel(downsampler_net, [0])
        upscale_net = nn.DataParallel(upscale_net, [0])

        kernel_generation_net.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, "{0}x".format(self.SCALE), "kgn.pth")
            )
        )
        upscale_net.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, "{0}x".format(self.SCALE), "usn.pth")
            )
        )
        # torch.set_grad_enabled(False)

        # self.networks
        self.kernel_generation_net = kernel_generation_net
        self.downsampler_net = downsampler_net
        self.upscale_net = upscale_net

        self.kernel_generation_net.eval()
        self.downsampler_net.eval()
        # self.upscale_net.eval()

    def downscale(self, img):
        with torch.no_grad():
            # model evaluation settings
            self.kernel_generation_net.eval()
            self.downsampler_net.eval()
            # upscale_net.eval()

            # networks
            img = img.unsqueeze(0)
            kernels, offsets_h, offsets_v = self.kernel_generation_net(img)
            downscaled_img = self.downsampler_net(
                img, kernels, offsets_h, offsets_v, self.OFFSET_UNIT
            )
            downscaled_img = torch.clamp(downscaled_img, 0, 1)
            # downscaled_img = torch.round(downscaled_img * 255)
            # reconstructed_img = upscale_net(downscaled_img / 255.0)

            # save image for checking
            """
            img_idx=0
            while(True):
                if os.path.exists("test_images/%d.png" % (img_idx)):
                    img_idx += 1
                else:
                    save_image(img[0], "test_images/%d.png" % (img_idx))
                    save_image(downscaled_img[0], "test_images/%d_.png" % (img_idx))
                    break
            """
            return downscaled_img.squeeze(0)


if __name__ == "__main__":
    PDSN = PostDownscaleNetwork()
    PDSN.init_network()
