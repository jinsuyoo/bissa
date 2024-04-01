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


class VideoTestDataset(data.Dataset):
    """Video test dataset.

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt

        if self.opt["name"] == "Vid4":
            self.categories = ["Calendar", "City", "Foliage", "Walk"]
            a, b, c, d = 41, 34, 49, 47
            self.clip_idx = [
                [0, a],
                [a, a + b],
                [a + b, a + b + c],
                [a + b + c, a + b + c + d],
            ]
        elif self.opt["name"] == "REDS4":
            self.categories = ["Clip_000", "Clip_011", "Clip_015", "Clip_020"]
            a, b, c, d = 100, 100, 100, 100
            self.clip_idx = [
                [0, a],
                [a, a + b],
                [a + b, a + b + c],
                [a + b + c, a + b + c + d],
            ]
        elif self.opt["name"] == "VideoLQ":
            self.categories = [str(i).zfill(3) for i in range(50)]
            # a, b, c, d = 100, 100, 100, 100
            # self.clip_idx = [[0, a], [a, a+b], [a+b, a+b+c], [a+b+c, a+b+c+d]]
            self.clip_idx = [[i, i + 100] for i in range(0, 5000, 100)]
        elif self.opt["name"] == "davis_testdev":
            self.categories = sorted(
                os.listdir("../../datasets/davis_testdev/GT")
            )
            self.frames_per_clip = [
                len(
                    glob.glob(
                        osp.join(
                            "../../datasets/davis_testdev/GT", cate, "*.png"
                        )
                    )
                )
                for cate in self.categories
            ]
            self.clip_idx = []
            acc_num = 0
            for i in range(len(self.categories)):
                self.clip_idx.append([acc_num, acc_num + self.frames_per_clip[i]])
                acc_num += self.frames_per_clip[i]
        # else:
        #    raise ValueError('dataset name error')

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

        # logger = get_root_logger()
        # logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        print(f'Generate data info for VideoTestDataset - {opt["name"]}')

        self.imgs_lq, self.imgs_gt = {}, {}
        if "meta_info_file" in opt:
            with open(opt["meta_info_file"], "r") as fin:
                subfolders = [line.split(" ")[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, "*")))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, "*")))

        if opt["name"].lower() in [
            "vid4",
            "reds4",
            "redsofficial",
            "videolq, davis_testdev",
        ]:
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


class VideoTestRBPNDataset(data.Dataset):
    def __init__(self, opt):
        super(VideoTestRBPNDataset, self).__init__()
        self.opt = opt

        alist = [
            line.rstrip() for line in open(osp.join(opt["image_dir"], opt["file_list"]))
        ]
        self.image_filenames = [osp.join(opt["image_dir"], x) for x in alist]
        self.nFrames = opt["n_frame"]
        self.upscale_factor = opt["upscale_factor"]
        self.transform = Compose([ToTensor(),])
        self.future_frame = opt["future_frame"]

        self.categories = [opt["name"]]
        self.clip_idx = [[0, len(self.image_filenames)]]

    def load_img_future(self, filepath, nFrames, scale):
        tt = int(nFrames / 2)

        target = self.modcrop(Image.open(filepath).convert("RGB"), scale)
        input = target.resize(
            (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC
        )

        char_len = len(filepath)
        neigbor = []

        seq = [x for x in range(-tt, tt + 1) if x != 0]
        print(filepath)
        print("seq", seq)
        # random.shuffle(seq) #if random sequence
        for i in seq:
            index1 = int(filepath[char_len - 7 : char_len - 4]) + i
            file_name1 = filepath[0 : char_len - 7] + "{0:03d}".format(index1) + ".png"
            print("filename1", file_name1)
            if osp.exists(file_name1):
                temp = self.modcrop(
                    Image.open(file_name1).convert("RGB"), scale
                ).resize(
                    (int(target.size[0] / scale), int(target.size[1] / scale)),
                    Image.BICUBIC,
                )
                neigbor.append(temp)
            else:
                # print('neigbor frame- is not exist')
                temp = input
                neigbor.append(temp)

        return target, input, neigbor

    def get_flow(self, im1, im2):
        im1 = np.array(im1)
        im2 = np.array(im2)
        im1 = im1.astype(float) / 255.0
        im2 = im2.astype(float) / 255.0

        # Flow Options:
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

        u, v, im2W = pyflow.coarse2fine_flow(
            im1,
            im2,
            alpha,
            ratio,
            minWidth,
            nOuterFPIterations,
            nInnerFPIterations,
            nSORIterations,
            colType,
        )
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        return flow

    def modcrop(self, img, modulo):
        (ih, iw) = img.size
        ih = ih - (ih % modulo)
        iw = iw - (iw % modulo)
        img = img.crop((0, 0, ih, iw))
        return img

    def __getitem__(self, index):
        # target : pil
        # input : pil
        # neighbor : list(pil, pil, ...)
        target, input, neigbor = self.load_img_future(
            self.image_filenames[index], self.nFrames, self.upscale_factor
        )

        flow = [self.get_flow(input, j) for j in neigbor]

        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            neigbor = [self.transform(j) for j in neigbor]
            flow = [torch.from_numpy(j.transpose(2, 0, 1)) for j in flow]

        return {
            "lq": input,
            "gt": target,
            "neighbor": neigbor,
            "flow": flow,
            "folder": self.opt["file_list"][:-4],
            "lq_path": self.image_filenames[index],  # center frame
        }
        # return input, target, neigbor, flow

    def __len__(self):
        return len(self.image_filenames)


class VideoTestRBPNDataset2(VideoTestDataset):
    def __init__(self, opt):
        super(VideoTestRBPNDataset2, self).__init__(opt)
        self.nFrames = opt["num_frame"]
        self.upscale_factor = opt["upscale_factor"]
        self.transform = Compose([ToTensor(),])
        self.future_frame = opt["future_frame"]

    def load_img(self, filepaths):
        targets = [self.modcrop(Image.open(p).convert("RGB"), 4) for p in filepaths]
        # inputs = target[self.opt['n_seq']//2].resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        inputs = [
            t.resize((int(t.size[0] / 4), int(t.size[1] / 4)), Image.BICUBIC)
            for t in targets
        ]
        target = targets[self.nFrames // 2]
        input = inputs[self.nFrames // 2]
        neighbor = []
        neighbor += inputs[: self.nFrames // 2]
        neighbor += inputs[(self.nFrames // 2) + 1 :]
        return target, input, neighbor

    def load_img_future(self, filepath, nFrames, scale):
        tt = int(nFrames / 2)

        target = self.modcrop(Image.open(filepath).convert("RGB"), scale)
        input = target.resize(
            (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC
        )

        char_len = len(filepath)
        neigbor = []

        seq = [x for x in range(-tt, tt + 1) if x != 0]
        print(filepath)
        print("seq", seq)
        # random.shuffle(seq) #if random sequence
        for i in seq:
            index1 = int(filepath[char_len - 7 : char_len - 4]) + i
            file_name1 = filepath[0 : char_len - 7] + "{0:03d}".format(index1) + ".png"
            print("filename1", file_name1)
            if osp.exists(file_name1):
                temp = self.modcrop(
                    Image.open(file_name1).convert("RGB"), scale
                ).resize(
                    (int(target.size[0] / scale), int(target.size[1] / scale)),
                    Image.BICUBIC,
                )
                neigbor.append(temp)
            else:
                # print('neigbor frame- is not exist')
                temp = input
                neigbor.append(temp)

        return target, input, neigbor

    def get_flow(self, im1, im2):
        im1 = np.array(im1)
        im2 = np.array(im2)
        im1 = im1.astype(float) / 255.0
        im2 = im2.astype(float) / 255.0

        # Flow Options:
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

        u, v, im2W = pyflow.coarse2fine_flow(
            im1,
            im2,
            alpha,
            ratio,
            minWidth,
            nOuterFPIterations,
            nInnerFPIterations,
            nSORIterations,
            colType,
        )
        flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        return flow

    def modcrop(self, img, modulo):
        (ih, iw) = img.size
        ih = ih - (ih % modulo)
        iw = iw - (iw % modulo)
        img = img.crop((0, 0, ih, iw))
        return img

    def __getitem__(self, index):
        folder = self.data_info["folder"][index]
        idx, max_idx = self.data_info["idx"][index].split("/")
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info["border"][index]
        lq_path = self.data_info["lq_path"][index]

        select_idx = util.generate_frame_indices(
            idx, max_idx, self.opt["num_frame"], padding=self.opt["padding"]
        )

        img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
        # print(img_paths_lq)
        # print([self.imgs_gt[folder][idx]])
        # imgs_lq = util.read_img_seq(img_paths_lq)
        # img_gt = util.read_img_seq([self.imgs_gt[folder][idx]])
        # img_gt.squeeze_(0)

        target, input, neighbor = self.load_img(img_paths_lq)
        # print(target, input, neighbor)

        flow = [self.get_flow(input, j) for j in neighbor]

        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            neighbor = [self.transform(j) for j in neighbor]
            flow = [torch.from_numpy(j.transpose(2, 0, 1)) for j in flow]

        return {
            "lq": input,
            "gt": target,
            "neighbor": neighbor,
            "flow": flow,
            "folder": folder,  # folder name
            "idx": self.data_info["idx"][index],  # e.g., 0/99
            "border": border,  # 1 for border, 0 for non-border
            "lq_path": lq_path,  # center frame
        }
        # return input, target, neighbor, flow
