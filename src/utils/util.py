import argparse
import numpy as np
import os
import random
import time
import torch
from os import path as osp

from .dist_util import master_only, get_dist_info, init_dist
from .options import parse


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt", type=str, required=True, help="Path to option YAML file."
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == "none":
        opt["dist"] = False
        print("Disable distributed.", flush=True)
    else:
        opt["dist"] = True
        if args.launcher == "slurm" and "dist_params" in opt:
            init_dist(args.launcher, **opt["dist_params"])
        else:
            init_dist(args.launcher)

    opt["rank"], opt["world_size"] = get_dist_info()

    # random seed
    seed = opt.get("manual_seed")
    if seed is None:
        seed = random.randint(1, 10000)
        opt["manual_seed"] = seed
    set_random_seed(seed + opt["rank"])

    return opt


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + "_archived_" + get_time_str()
        print(f"Path already exists. Renamed it to {new_name}", flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


@master_only
def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt["path"].copy()
    if opt["is_train"]:
        mkdir_and_rename(path_opt.pop("experiments_root"))
    else:
        mkdir_and_rename(path_opt.pop("results_root"))
    for key, path in path_opt.items():
        if (
            ("strict_load" not in key)
            and ("pretrain_network" not in key)
            and ("resume" not in key)
        ):
            os.makedirs(path, exist_ok=True)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative patches.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def sizeof_fmt(size, suffix="B"):
    """Get human readable file size.

    Args:
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
        str: Formatted file size.
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}{suffix}"
        size /= 1024.0
    return f"{size:3.1f} Y{suffix}"
