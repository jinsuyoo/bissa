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


class Img2VidDataset(data.Dataset):
    def __init__(self, opt):
        super(Img2VidDataset, self).__init__()
        self.opt = opt

        if self.opt["name"] == "DIV2K+Urban100":
            self.categories = ["Total"]
            # self.clip_idx = [[0, a], [a, a+b], [a+b, a+b+c]]
            # self.clip_idx = [[0, a], [0, b], [0, c]]
            self.clip_idx = [[0, 900]]

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

        print(f'Generate data info for Img2VidDataset - {opt["name"]}')

        self.imgs_lq, self.imgs_gt = {}, {}
        if "meta_info_file" in opt:
            with open(opt["meta_info_file"], "r") as fin:
                subfolders = [line.split(" ")[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, "*")))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, "*")))

        if opt["name"].lower() in ["div2k+urban100"]:
            for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
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

                # No border on image2vid dataset
                border_l = [0] * max_idx
                self.data_info["border"].extend(border_l)

                # cache data or save the frame list
                if self.cache_data:
                    logger.info(f"Cache {subfolder_name} for Img2VidDataset...")
                    self.imgs_lq[subfolder_name] = util.read_img_seq(img_paths_lq)
                    self.imgs_gt[subfolder_name] = util.read_img_seq(img_paths_gt)
                else:
                    self.imgs_lq[subfolder_name] = img_paths_lq
                    self.imgs_gt[subfolder_name] = img_paths_gt
        else:
            raise ValueError(f'Non-supported video test dataset: {type(opt["name"])}')

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
            imgs_lq = torch.stack([self.imgs_lq[folder][idx]] * len(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][idx]] * len(select_idx)
            imgs_lq = util.read_img_seq(img_paths_lq)  # [:,:,:300,:300]
            img_gt = util.read_img_seq([self.imgs_gt[folder][idx]])  # [:,:,:1200,:1200]
            # print(imgs_lq.size(), img_gt.size())
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
