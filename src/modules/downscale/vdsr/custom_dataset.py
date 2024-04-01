import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils import data as data
from vdsr import Net
import random
from PIL import Image

from custom_util import imresize


class CustomDataset(data.Dataset):
    def __init__(self):
        super(CustomDataset, self).__init__()
        # self.lq_root = "DIV2K/train/P_GT"
        # self.gt_root = "DIV2K/train/GT"
        self.lq_root = "DIV2K/scales/P_GT"
        self.gt_root = "DIV2K/scales/GT"
        self.tft = transforms.ToTensor()
        self.input_w = 256
        self.input_h = 256

        self.data = [
            (
                os.path.join(self.lq_root, image_folder),
                os.path.join(self.gt_root, image_folder),
            )
            for image_folder in os.listdir(self.lq_root)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # random scale
        random_scale = random.randint(80, 95) / 100
        lq_path, gt_path = self.data[index]
        lq_path = os.path.join(lq_path, f"{random_scale:.2f}" + ".png")
        gt_path = os.path.join(gt_path, f"{random_scale:.2f}" + ".png")
        lq = Image.open(lq_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        lq = self.tft(lq)
        gt = self.tft(gt)

        # random crop
        _, h, w = lq.shape
        start_x = random.randint(0, w - self.input_w)
        start_y = random.randint(0, h - self.input_h)
        lq = lq[:, start_y : start_y + self.input_h, start_x : start_x + self.input_w]
        gt = gt[:, start_y : start_y + self.input_h, start_x : start_x + self.input_w]

        # lq, gt = self.flip_and_rotate(lq, gt)
        return lq, gt

    def flip_and_rotate(self, lq, gt):
        # flip
        flip = random.randrange(0, 3)
        if flip != 0:
            lq = torch.flip(lq, dims=[flip])
            gt = torch.flip(gt, dims=[flip])

        # rotate
        degree = random.randrange(0, 4)
        lq = torch.rot90(lq, degree, [1, 2])
        gt = torch.rot90(gt, degree, [1, 2])

        return lq, gt

    # patch version
    def preprocess(self, lq, gt):
        # lq, gt = lq.cuda(), gt.cuda()

        # downsclae with random scale (0.80 to 0.95)
        scale = random.randrange(80, 100) / 100
        lq = imresize(lq, scale)
        gt = imresize(gt, scale)

        # 0:None, 1:상하반전, 2:좌우반전
        flip = random.randrange(0, 3)
        if flip != 0:
            lq = torch.flip(lq, dims=[flip])
            gt = torch.flip(gt, dims=[flip])

        # degree*90 rotate
        degree = random.randrange(0, 4)
        lq = torch.rot90(lq, degree, [1, 2])
        gt = torch.rot90(gt, degree, [1, 2])

        # conversion
        lq = rgb_to_ycbcr(lq)[0, :, :]
        lq = torch.unsqueeze(lq, 0)
        gt = rgb_to_ycbcr(gt)[0, :, :]
        gt = torch.unsqueeze(gt, 0)

        # random crop
        _, h, w = lq.shape
        input_w = min(w, self.input_w)
        input_h = min(h, self.input_h)
        x = random.randrange(0, w - input_w + 1)
        y = random.randrange(0, h - input_h + 1)
        lq = lq[:, y : y + input_h, x : x + input_w]
        gt = gt[:, y : y + input_h, x : x + input_w]

        return lq, gt


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    .. image:: _static/img/rgb_to_ycbcr.png

    Args:
        image: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        YCbCr version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_ycbcr(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"
        )

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    cb: torch.Tensor = (b - y) * 0.564 + delta
    cr: torch.Tensor = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3)
