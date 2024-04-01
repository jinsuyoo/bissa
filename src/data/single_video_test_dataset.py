import glob
import torch
from os import path as osp

import numpy as np
from PIL import Image
from torch.utils import data as data
from torchvision.transforms import Compose, ToTensor

from src.data import util as util
from src.data.util import duf_downsample, generate_gaussian_kernel
from src.utils import scandir, pyflow


class SingleVideoTestDataset(data.Dataset):
    def __init__(self, opt):
        super(SingleVideoTestDataset, self).__init__()
        self.opt = opt

        self.cache_data = opt["cache_data"]
        self.gt_root, self.lq_root = opt["dataroot_gt"], opt["dataroot_lq"]
        self.data_info = {
            "lq_path": [],
            "gt_path": [],
            "folder": [],
            "idx": [],
            "border": [],
        }
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        assert (
            self.io_backend_opt["type"] != "lmdb"
        ), "No need to use lmdb during validation/test."

        print(f'Generate data info for VideoTestDataset - {opt["name"]}')

        self.imgs_lq, self.imgs_gt = {}, {}

        # subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
        # subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        subfolder_lq = self.lq_root
        subfolder_gt = self.gt_root

        # get frame list for lq and gt
        subfolder_name = osp.basename(subfolder_lq)
        img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))
        img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))

        max_idx = len(img_paths_lq)
        assert max_idx == len(img_paths_gt), (
            f"Different number of images in lq ({max_idx})"
            f" and gt folders ({len(img_paths_gt)})"
        )

        self.data_info["lq_path"].extend(img_paths_lq)
        self.data_info["gt_path"].extend(img_paths_gt)
        self.data_info["folder"].extend([subfolder_name] * max_idx)
        for i in range(max_idx):
            self.data_info["idx"].append(f"{i}/{max_idx}")
        border_l = [0] * max_idx
        for i in range(self.opt["num_frame"] // 2):
            border_l[i] = 1
            border_l[max_idx - i - 1] = 1
        self.data_info["border"].extend(border_l)

        # cache data or save the frame list
        if self.cache_data:
            print(f"Cache {subfolder_name} for VideoTestDataset...")
            self.imgs_lq[subfolder_name] = util.read_img_seq(img_paths_lq)
            self.imgs_gt[subfolder_name] = util.read_img_seq(img_paths_gt)
        else:
            self.imgs_lq[subfolder_name] = img_paths_lq
            self.imgs_gt[subfolder_name] = img_paths_gt

    def __getitem__(self, index):
        folder = self.data_info["folder"][index]
        idx, max_idx = self.data_info["idx"][index].split("/")
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info["border"][index]
        lq_path = self.data_info["lq_path"][index]

        select_idx = util.generate_frame_indices(
            idx, max_idx, self.opt["num_frame"], padding=self.opt["padding"]
        )

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = util.read_img_seq(img_paths_lq)
            img_gt = util.read_img_seq([self.imgs_gt[folder][idx]])
            img_gt.squeeze_(0)

        return {
            "lq": imgs_lq,  # (t, c, h, w)
            "gt": img_gt,  # (c, h, w)
            "folder": folder,  # folder name
            "idx": self.data_info["idx"][index],  # e.g., 0/99
            "border": border,  # 1 for border, 0 for non-border
            "lq_path": lq_path,  # center frame
        }

    def __len__(self):
        return len(self.data_info["gt_path"])
