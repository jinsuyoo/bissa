import argparse, os
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument(
    "--model", default="model/official_pretrained.pth", type=str, help="model path"
)
parser.add_argument("--dataset", default="P_GT", type=str, help="")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[
        shave_border : height - shave_border, shave_border : width - shave_border
    ]
    gt = gt[shave_border : height - shave_border, shave_border : width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


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


tft = transforms.ToTensor()

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
            im_gt = tft(im_gt)
            _, h, w = im_gt.size()
            im_gt_y = rgb_to_ycbcr(im_gt)[0, :h, :w] * 255
            im_gt_y = im_gt_y.numpy()

            im_b = Image.open(os.path.join(lq_root, image_name))
            im_b = tft(im_b)
            im_b_y = rgb_to_ycbcr(im_b)[0, :h, :w] * 255
            im_b_y = im_b_y.numpy()

            im_gt_y = im_gt_y.astype(float)
            im_b_y = im_b_y.astype(float)

            scale = 2
            psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=scale)
            avg_psnr_bicubic += psnr_bicubic

            im_input = im_b_y / 255.0

            im_input = Variable(torch.from_numpy(im_input).float()).view(
                1, -1, im_input.shape[0], im_input.shape[1]
            )

            if cuda:
                model = model.cuda()
                im_input = im_input.cuda()
            else:
                model = model.cpu()

            start_time = time.time()
            with torch.no_grad():
                HR = model(im_input)
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

            HR = HR.cpu()

            im_h_y = HR.data[0].numpy().astype(np.float32)

            im_h_y = im_h_y * 255.0
            im_h_y[im_h_y < 0] = 0
            im_h_y[im_h_y > 255.0] = 255.0
            im_h_y = im_h_y[0, :, :]

            scale = 2
            psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=scale)
            avg_psnr_predicted += psnr_predicted

    print("Dataset=", opt.dataset)
    print("PSNR_predicted=", avg_psnr_predicted / count)
    print("PSNR_bicubic=", avg_psnr_bicubic / count)
    print("It takes average {}s for processing".format(avg_elapsed_time / count))
