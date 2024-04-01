import argparse, os
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
from PIL import Image
from tqdm import tqdm

from custom_metrics import calculate_psnr, calculate_ssim, tensor2img


parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument(
    "--model", default="model/patch_epoch_23.pth", type=str, help="model path"
)
parser.add_argument("--dataset", default="P_GT", type=str, help="")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")


def calculate_metric(img1, img2, y_channel=True):
    psnr = calculate_psnr(
        img1, img2, crop_border=0, input_order="HWC", test_y_channel=True
    )
    ssim = calculate_ssim(
        img1, img2, crop_border=0, input_order="HWC", test_y_channel=True
    )

    return psnr, ssim


tft = transforms.ToTensor()
tfp = transforms.ToPILImage()
opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
model = model.eval()

if True:
    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    avg_ssim_predicted = 0.0
    avg_ssim_bicubic = 0.0
    avg_elapsed_time = 0.0
    count = 0.0

    gt_root = "DIV2K/valid/GT"
    lq_root = "DIV2K/valid/P_GT"
    if opt.dataset == "BIx4":
        gt_root = "DIV2K/valid_same/GT"
        lq_root = "DIV2K/valid_same/BIx4"

    image_list = os.listdir(gt_root)
    for image_name in tqdm(image_list):
        if True:
            count += 1
            # print("Processing ", image_name)
            im_gt = Image.open(os.path.join(gt_root, image_name))
            im_lq = Image.open(os.path.join(lq_root, image_name))

            im_gt = tft(im_gt)
            im_lq = tft(im_lq)

            _, h, w = im_gt.size()
            print(_, h, w, im_lq.size())

            im_gt = im_gt[:, :h, :w]
            im_lq = im_lq[:, :h, :w]

            psnr_bicubic, ssim_bicubic = calculate_metric(
                tensor2img(im_gt), tensor2img(im_lq)
            )
            avg_psnr_bicubic += psnr_bicubic
            avg_ssim_bicubic += ssim_bicubic

            if cuda:
                model = model.cuda()
                im_lq = im_lq.cuda()
            else:
                model = model.cpu()

            start_time = time.time()
            with torch.no_grad():
                im_hq = model(im_lq.unsqueeze(0))
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

            im_hq = im_hq.squeeze(0)
            im_hq = im_hq.cpu()

            psnr_predicted, ssim_predicted = calculate_metric(
                tensor2img(im_gt), tensor2img(im_hq)
            )
            avg_psnr_predicted += psnr_predicted
            avg_ssim_predicted += ssim_predicted

    print("Dataset=", opt.dataset)
    print("PSNR_predicted=", avg_psnr_predicted / count)
    print("PSNR_bicubic=", avg_psnr_bicubic / count)
    print("SSIM_predicted=", avg_ssim_predicted / count)
    print("SSIM_bicubic=", avg_ssim_bicubic / count)
    print("It takes average {}s for processing".format(avg_elapsed_time / count))
