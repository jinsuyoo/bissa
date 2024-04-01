import random

import glob
import torch
from os import path as osp

import numpy as np
from PIL import Image
from torch.utils import data as data
from torchvision.transforms import Compose, ToTensor

from src.data import util as util
from src.utils import scandir, imresize


class VideoAdaptDataset(data.Dataset):
    def __init__(self, opt, downscale=True):
        super(VideoAdaptDataset, self).__init__()
        self.opt = opt

        self.downscale = downscale

        self.cache_data = opt["cache_data"]
        self.base_root = opt["dataroot_base"]

        print(f'Generate data info for VideoAdaptDataset - {opt["name"]}')

        subfolder_base = self.base_root

        # get frame list for lq and gt
        img_paths_base = sorted(list(scandir(subfolder_base, full_path=True)))

        # cache data or save the frame list
        if self.cache_data:
            print(f"Cache data for VideoAdaptDataset...")
            # Tensor: size (t, c, h, w), RGB, [0, 1].
            self.imgs_base = util.read_img_seq(img_paths_base)

        else:
            self.imgs_base = img_paths_base

        self.n_frame = len(img_paths_base)

    def get_random_frame(self):
        current_frame = random.randint(0, self.n_frame - 1)

        frame_idx = self.generate_frame_indices(
            current_frame, self.n_frame, self.opt["n_seq"], self.opt["padding"]
        )

        if (
            "random_reverse" in self.opt
            and self.opt["random_reverse"] is True
            and np.random.random() >= 0.5
        ):
            frame_idx.reverse()

        pseudo_target = torch.index_select(self.imgs_base, 0, torch.tensor(frame_idx))

        return pseudo_target, frame_idx

    def generate_frame_indices(
        self, crt_idx, max_frame_num, num_frames, padding="reflection"
    ):
        max_frame_num = max_frame_num - 1  # start from 0
        num_pad = num_frames // 2

        indices = []
        for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
            if i < 0:
                if padding == "replicate":
                    pad_idx = 0
                elif padding == "reflection":
                    pad_idx = -i
                elif padding == "reflection_circle":
                    pad_idx = crt_idx + num_pad - i
                else:
                    pad_idx = num_frames + i
            elif i > max_frame_num:
                if padding == "replicate":
                    pad_idx = max_frame_num
                elif padding == "reflection":
                    pad_idx = max_frame_num * 2 - i
                elif padding == "reflection_circle":
                    pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
                else:
                    pad_idx = i - num_frames
            else:
                pad_idx = i
            indices.append(pad_idx)
        return indices

    def random_resize(self, pseudo_target):
        if self.downscale:
            random_scale = (
                random.randint(
                    int(self.opt["min_scale"] * 100), int(self.opt["max_scale"] * 100)
                )
                / 100
            )
            pseudo_target = imresize(pseudo_target, scale=random_scale)
        return pseudo_target

    def random_crop(self, pseudo_target):
        _, _, h, w = pseudo_target.shape
        # random crop
        x = random.randrange(0, w - self.opt["patch_size"] + 1)
        y = random.randrange(0, h - self.opt["patch_size"] + 1)
        pseudo_target = pseudo_target[
            ..., y : y + self.opt["patch_size"], x : x + self.opt["patch_size"]
        ]

        return pseudo_target

    def generate_pseudo_input(self, pseudo_target):
        # return torch.stack(
        #    [imresize(p, scale=0.25) for p in pseudo_target]
        # )
        return imresize(pseudo_target, scale=0.25)
        # return resize_right.resize(pseudo_target, scale_factors=0.25)

    def __getitem__(self, idx):
        pseudo_target, frame_idx = self.get_random_frame()
        pseudo_target = self.random_resize(pseudo_target)
        pseudo_target = self.random_crop(pseudo_target)
        pseudo_input = self.generate_pseudo_input(pseudo_target)

        return {
            "lq": pseudo_input,
            "gt": pseudo_target[self.opt["n_seq"] // 2, ...],
            "frame_idx": frame_idx,
        }

    def __len__(self):
        return 1000000
