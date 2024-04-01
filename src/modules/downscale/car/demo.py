import os, argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image

import torch
import torch.nn as nn

import utils
from EDSR.edsr import EDSR
from modules import DSN
from adaptive_gridsampler.gridsampler import Downsampler
from skimage.color import rgb2ycbcr
from imresize import imresize

parser = argparse.ArgumentParser(
    description="Content Adaptive Resampler for Image downscaling"
)
parser.add_argument(
    "--model_dir", type=str, default="./models", help="path to the pre-trained model"
)
parser.add_argument(
    "--img_dir",
    type=str,
    default="./images",
    help="path to the HR images to be downscaled",
)
parser.add_argument("--scale", type=int, default=4, help="downscale factor")
parser.add_argument(
    "--output_dir", type=str, default="./results", help="path to store results"
)
parser.add_argument(
    "--benchmark", type=bool, default=True, help="report benchmark results"
)
args = parser.parse_args()


SCALE = args.scale
KSIZE = 3 * SCALE + 1
OFFSET_UNIT = SCALE
BENCHMARK = args.benchmark

kernel_generation_net = DSN(k_size=KSIZE, scale=SCALE).cuda()
downsampler_net = Downsampler(SCALE, KSIZE).cuda()
upscale_net = EDSR(32, 256, scale=SCALE).cuda()

kernel_generation_net = nn.DataParallel(kernel_generation_net, [0])
downsampler_net = nn.DataParallel(downsampler_net, [0])
upscale_net = nn.DataParallel(upscale_net, [0])

kernel_generation_net.load_state_dict(
    torch.load(os.path.join(args.model_dir, "{0}x".format(SCALE), "kgn.pth"))
)
upscale_net.load_state_dict(
    torch.load(os.path.join(args.model_dir, "{0}x".format(SCALE), "usn.pth"))
)
torch.set_grad_enabled(False)


def validation(img, name, save_imgs=False, save_dir=None):

    # kernel_generation_net : 0-1 input
    # downsampler_net : 0-1 input, 0-1~ output
    kernel_generation_net.eval()
    downsampler_net.eval()
    upscale_net.eval()

    kernels, offsets_h, offsets_v = kernel_generation_net(img)

    # imresize function
    resized_img = imresize(img, scale=0.25)
    resized_img = torch.clamp(resized_img, 0, 1)

    # downscale network
    downscaled_img = downsampler_net(img, kernels, offsets_h, offsets_v, OFFSET_UNIT)
    downscaled_img = torch.clamp(downscaled_img, 0, 1)

    # recon images
    recon_img_d = upscale_net(downscaled_img)
    recon_img_r = upscale_net(resized_img)
    recon_img_d = torch.clamp(recon_img_d, 0, 1)
    recon_img_r = torch.clamp(recon_img_r, 0, 1)

    # Show Results
    img = torch.round(img * 255)
    resized_img = torch.round(resized_img * 255)
    downscaled_img = torch.round(downscaled_img * 255)
    recon_img_d = torch.round(recon_img_d * 255)
    recon_img_r = torch.round(recon_img_r * 255)

    img = img.data.cpu().numpy().transpose(0, 2, 3, 1)
    resized_img = resized_img.data.cpu().numpy().transpose(0, 2, 3, 1)
    downscaled_img = downscaled_img.data.cpu().numpy().transpose(0, 2, 3, 1)
    recon_img_d = recon_img_d.data.cpu().numpy().transpose(0, 2, 3, 1)
    recon_img_r = recon_img_r.data.cpu().numpy().transpose(0, 2, 3, 1)

    img = np.uint8(img)
    resized_img = np.uint8(resized_img)
    downscaled_img = np.uint8(downscaled_img)
    recon_img_d = np.uint8(recon_img_d)
    recon_img_r = np.uint8(recon_img_r)

    orig_img = img[0, ...].squeeze()
    resized_img = resized_img[0, ...].squeeze()
    downscaled_img = downscaled_img[0, ...].squeeze()
    recon_img_d = recon_img_d[0, ...].squeeze()
    recon_img_r = recon_img_r[0, ...].squeeze()

    print(
        "mean original\t: ",
        type(orig_img),
        orig_img.shape,
        np.mean(orig_img),
        np.std(orig_img),
    )
    print(
        "mean resized\t: ",
        type(resized_img),
        resized_img.shape,
        np.mean(resized_img),
        np.std(resized_img),
    )
    print(
        "mean downscaled\t: ",
        type(downscaled_img),
        downscaled_img.shape,
        np.mean(downscaled_img),
        np.std(downscaled_img),
    )
    print(
        "mean recon_d\t: ",
        type(recon_img_d),
        recon_img_d.shape,
        np.mean(recon_img_d),
        np.std(recon_img_d),
    )
    print(
        "mean recon_r\t: ",
        type(recon_img_r),
        recon_img_r.shape,
        np.mean(recon_img_r),
        np.std(recon_img_r),
    )

    if save_imgs and save_dir:
        img = Image.fromarray(orig_img)
        img.save(os.path.join(save_dir, name + "_orig.png"))

        img = Image.fromarray(downscaled_img)
        img.save(os.path.join(save_dir, name + "_down.png"))

        img = Image.fromarray(resized_img)
        img.save(os.path.join(save_dir, name + "_resized.png"))

        img = Image.fromarray(recon_img_d)
        img.save(os.path.join(save_dir, name + "_recon_d.png"))

        img = Image.fromarray(recon_img_r)
        img.save(os.path.join(save_dir, name + "_recon_r.png"))

    psnr_d = utils.cal_psnr(
        orig_img[SCALE:-SCALE, SCALE:-SCALE, ...],
        recon_img_d[SCALE:-SCALE, SCALE:-SCALE, ...],
        benchmark=BENCHMARK,
    )
    psnr_r = utils.cal_psnr(
        orig_img[SCALE:-SCALE, SCALE:-SCALE, ...],
        recon_img_r[SCALE:-SCALE, SCALE:-SCALE, ...],
        benchmark=BENCHMARK,
    )

    orig_img_y = rgb2ycbcr(orig_img)[:, :, 0]
    recon_img_d_y = rgb2ycbcr(recon_img_d)[:, :, 0]
    recon_img_r_y = rgb2ycbcr(recon_img_r)[:, :, 0]
    orig_img_y = orig_img_y[SCALE:-SCALE, SCALE:-SCALE, ...]
    recon_img_d_y = recon_img_d_y[SCALE:-SCALE, SCALE:-SCALE, ...]
    recon_img_r_y = recon_img_r_y[SCALE:-SCALE, SCALE:-SCALE, ...]

    ssim_d = utils.calc_ssim(recon_img_d_y, orig_img_y)
    ssim_r = utils.calc_ssim(recon_img_r_y, orig_img_y)

    return psnr_d, ssim_d, psnr_r, ssim_r


if __name__ == "__main__":
    img_list = glob(os.path.join(args.img_dir, "**", "*.png"), recursive=True)
    assert len(img_list) > 0

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    psnr_d_list = list()
    ssim_d_list = list()
    psnr_r_list = list()
    ssim_r_list = list()
    for img_file in tqdm(img_list):
        name = os.path.basename(img_file)
        name = os.path.splitext(name)[0]

        img = utils.load_img(img_file)
        print("input shape: ", img.shape, img)
        psnr_d, ssim_d, psnr_r, ssim_r = validation(
            img, name, save_imgs=True, save_dir=args.output_dir
        )
        psnr_d_list.append(psnr_d)
        ssim_d_list.append(ssim_d)
        psnr_r_list.append(psnr_r)
        ssim_r_list.append(ssim_r)

    print("Mean PSNR_d: {0:.2f}".format(np.mean(psnr_d_list)))
    print("Mean SSIM_d: {0:.4f}".format(np.mean(ssim_d_list)))
    print("Mean PSNR_r: {0:.2f}".format(np.mean(psnr_r_list)))
    print("Mean SSIM_r: {0:.4f}".format(np.mean(ssim_r_list)))
