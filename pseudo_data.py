import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from utils import imresize
from utils import resize_right

try:
    from utils import pyflow
except:
    print(
        "[!] pyflow import failed. Just ignore this if the baseline model is not RBPN !"
    )


class PseudoDataset(data.Dataset):
    def __init__(self, initial_output, opt, length, downscale=True):
        super(PseudoDataset, self).__init__()
        self.opt = opt
        self.initial_output = initial_output
        self.n_frame = self.initial_output.shape[0]
        self.length = length
        self.downscale = downscale

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

        pseudo_target = torch.index_select(
            self.initial_output, 0, torch.tensor(frame_idx)
        )

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

            # pseudo_target = torch.stack([imresize(p, scale=random_scale) for p in pseudo_target])
            pseudo_target = imresize(pseudo_target, scale=random_scale)
            # pseudo_target = resize_right.resize(pseudo_target, scale_factors=random_scale)
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
        return self.length * self.opt["batch_size"]


class PseudoRBPNDataset(PseudoDataset):
    def __init__(self, initial_output, opt, length):
        super(PseudoRBPNDataset, self).__init__(initial_output, opt, length)

    def get_flow(self, im1, im2):
        im1 = transforms.ToPILImage()(im1)
        im2 = transforms.ToPILImage()(im2)
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

    def split_input(self, pseudo_input):
        input = pseudo_input[self.opt["n_seq"] // 2, ...]
        neighbor = pseudo_input[
            [i for i in range(self.opt["n_seq"]) if i is not self.opt["n_seq"] // 2],
            ...,
        ]

        return input, neighbor

    def random_resize(self, pseudo_target):
        def _generate(img, rs):
            img = transforms.ToPILImage()(img)
            width, height = int(img.width * rs), int(img.height * rs)
            img = img.resize((width, height), Image.BICUBIC)
            img = transforms.ToTensor()(img)
            return img

        random_scale = (
            random.randint(
                int(self.opt["min_scale"] * 100), int(self.opt["max_scale"] * 100)
            )
            / 100
        )

        pseudo_target = torch.stack([_generate(p, random_scale) for p in pseudo_target])

        return pseudo_target

    def generate_pseudo_input(self, pseudo_target):
        def _generate(img):
            img = transforms.ToPILImage()(img)
            img = img.resize(
                (self.opt["patch_size"] // 4, self.opt["patch_size"] // 4),
                Image.BICUBIC,
            )
            img = transforms.ToTensor()(img)
            return img

        pseudo_input = torch.stack([_generate(p) for p in pseudo_target])
        return pseudo_input

    def __getitem__(self, idx):
        pseudo_target, frame_idx = self.get_random_frame()
        pseudo_target = self.random_resize(pseudo_target)
        pseudo_target = self.random_crop(pseudo_target)
        pseudo_input = self.generate_pseudo_input(pseudo_target)

        pseudo_target = pseudo_target[self.opt["n_seq"] // 2, ...]
        pseudo_input, neighbor = self.split_input(pseudo_input)
        flow = [self.get_flow(pseudo_input, n) for n in neighbor]

        neighbor = [n for n in neighbor]
        flow = [torch.from_numpy(j.transpose(2, 0, 1)) for j in flow]

        return {
            "lq": pseudo_input,
            "gt": pseudo_target,
            "neighbor": neighbor,
            "flow": flow,
        }

    def __len__(self):
        return self.length * self.opt["batch_size"]


class PseudoBasicVSRDataset(PseudoDataset):
    def __init__(self, initial_output, opt, length, downscale):
        super(PseudoBasicVSRDataset, self).__init__(
            initial_output, opt, length, downscale
        )

    def __getitem__(self, idx):
        if self.opt["n_seq"] < 0:
            pseudo_target = self.initial_output
            frame_idx = list(range(self.n_frame))
        else:
            pseudo_target, frame_idx = self.get_random_frame()

        pseudo_target = self.random_resize(pseudo_target)
        pseudo_target = self.random_crop(pseudo_target)
        pseudo_input = self.generate_pseudo_input(pseudo_target)
        return {"lq": pseudo_input, "gt": pseudo_target, "frame_idx": frame_idx}


def create_pseudo_dataloader(opt, initial_output, length, downscale=True):
    if opt["model_type"] == "rbpn":
        dataset = PseudoRBPNDataset(initial_output, opt, length)
    elif opt["model_type"] == "basicvsr":
        dataset = PseudoBasicVSRDataset(
            initial_output, opt, length, downscale=downscale
        )
    else:
        dataset = PseudoDataset(initial_output, opt, length, downscale=downscale)

    return data.DataLoader(
        dataset=dataset, batch_size=opt["batch_size"], shuffle=False, num_workers=12
    )
