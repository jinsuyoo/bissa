import os
import itertools

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.utils.data as data

from src.data import create_dataloader, create_dataset
from src.models import create_model
from src.utils import make_exp_dirs, parse_options, Logger, tensor2img
from src.metrics import calculate_psnr, calculate_ssim
from src import losses

# parse options, set distributed setting, set random seed
opt = parse_options(is_train=True)

# mkdir and initialize loggers
make_exp_dirs(opt)
opt["save_dir"] = opt["path"]["visualization"]
logger = Logger(opt["path"]["log"], "log.txt")
logger.log_option(opt)

model = create_model(opt)
model.cuda()

if opt["evaluate"]:
    base_dataset_opt = list(opt["base_dataset"].values())[0]
    base_dataset = create_dataset(base_dataset_opt)
    base_dataloader = create_dataloader(
        base_dataset,
        base_dataset_opt,
        num_gpu=opt["num_gpu"],
        dist=opt["dist"],
        seed=opt["manual_seed"],
    )
    if len(base_dataloader) == 0:
        logger.log(f"Failed to load dataset {base_dataset_opt['name']}. Aborting.")
        exit(1)
    logger.log(
        f"Number of test images in {base_dataset_opt['name']}: "
        f"{len(base_dataset.data_info['gt_path'])}\n"
    )

adapt_dataset_opt = list(opt["adapt_dataset"].values())[0]
adapt_dataset = create_dataset(adapt_dataset_opt)
# adapt_dataloader = create_dataloader(
#    adapt_dataset, adapt_dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], seed=opt['manual_seed'])
adapt_dataloader = data.DataLoader(
    dataset=adapt_dataset, batch_size=opt["batch_size"], shuffle=False, num_workers=24
)
if len(adapt_dataloader) == 0:
    logger.log(f"Failed to load dataset {adapt_dataset_opt['name']}. Aborting.")
    exit(1)
logger.log(
    f"Number of test images in {adapt_dataset_opt['name']}: "
    f"{adapt_dataset.n_frame}\n"
)

folder = os.path.join(opt["save_dir"])
os.makedirs(folder + "/gt", exist_ok=True)
os.makedirs(folder + "/adapt", exist_ok=True)

criterion = losses.create_loss(opt["loss"]).cuda()
optimizer = optim.Adam(model.parameters(), lr=opt["learning_rate"])

model.train()

progress_bar = tqdm(total=opt["iterations"], desc="Adapting")

for i, data in enumerate(adapt_dataloader):
    if i == opt["iterations"]:
        break

    lr, gt = data["lq"], data["gt"]
    lr, gt = lr.cuda(), gt.cuda()

    hr = model(lr)

    loss = criterion(hr, gt)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    progress_bar.update()

if opt["evaluate"]:

    ### final evaluation
    model.eval()

    psnr_all = []
    ssim_all = []

    for i, data in enumerate(base_dataloader):
        lr, gt = data["lq"], data["gt"]
        lr, gt = lr.cuda(), gt.cuda()

        with torch.no_grad():
            hr = model(lr)

        gt = tensor2img(gt)
        hr = tensor2img(hr)

        psnr = calculate_psnr(
            gt, hr, crop_border=0, input_order="HWC", test_y_channel=True
        )
        ssim = calculate_ssim(
            gt, hr, crop_border=0, input_order="HWC", test_y_channel=True
        )

        psnr_all.append(psnr)
        ssim_all.append(ssim)

        # Save image.
        img_num = f"{i:08d}"
        gt_img_path = os.path.join(folder, "gt", f"{img_num}_gt.png")
        cv2.imwrite(gt_img_path, gt)

        hr_img_path = str(os.path.join(folder, "adapt", f"{img_num}_adapt_"))
        hr_img_path += f"psnr{int(100 * psnr)}_ssim{int(10000 * ssim)}.png"
        cv2.imwrite(hr_img_path, hr)

    # Print average metric value over a single dataset.
    psnr_all = sum(psnr_all) / len(psnr_all)
    ssim_all = sum(ssim_all) / len(ssim_all)

    logger.log(f"Adaptation PSNR/SSIM: {psnr_all:.6f}/{ssim_all:.6f}")
