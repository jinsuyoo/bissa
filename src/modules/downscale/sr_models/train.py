import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils import data as data
import random
from PIL import Image
from tqdm import tqdm

from dataset import Dataset
from models.edsr_model import Net as EDSR
from models.rcan_model import Net as RCAN
import warnings

warnings.filterwarnings(action="ignore")

# Training settings
parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument(
    "--nEpochs", type=int, default=50, help="Number of epochs to train for"
)
parser.add_argument("--lr", type=float, default=1e-1, help="Learning Rate. Default=0.1")
parser.add_argument(
    "--step",
    type=int,
    default=10,
    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10",
)
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument(
    "--resume", default="", type=str, help="Path to checkpoint (default: none)"
)
parser.add_argument(
    "--start-epoch",
    default=1,
    type=int,
    help="Manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4"
)
parser.add_argument(
    "--threads",
    type=int,
    default=16,
    help="Number of threads for data loader to use, Default: 1",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, help="Momentum, Default: 0.9"
)
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    help="Weight decay, Default: 1e-4",
)
parser.add_argument(
    "--pretrained",
    default="",
    type=str,
    help="path to pretrained model (default: none)",
)
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--lr-decay", type=float, default=0.5)
parser.add_argument("--checkpoint", type=str, default="checkpoint")
parser.add_argument("--model", type=str, default="edsr")
parser.add_argument("--scale", type=int, default=1)


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    if opt.model == "edsr":
        opt.batchSize = 2
    elif opt.model == "rcan":
        opt.batchSize = 1
    train_set = Dataset()
    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=True,
    )

    print("===> Building model")
    if opt.model == "edsr":
        opt.num_blocks = 32
        opt.num_channels = 256
        opt.res_scale = 0.1
        model = EDSR(opt)
    elif opt.model == "rcan":
        opt.num_groups = 10
        opt.num_blocks = 20
        opt.num_channels = 64
        opt.reduction = 16
        opt.res_scale = 1.0
        opt.max_steps = 1000000
        opt.decay = "200-400-600-800"
        opt.pretrain = False
        opt.gclip = 0  # base
        opt.gclip = 0.5 if opt.pretrain else opt.gclip
        model = RCAN(opt)
    elif opt.model == "CARN":
        opt.num_groups = 3
        opt.num_blocks = 3
        opt.num_channels = 64
        opt.res_scale = 1.0
        opt.batch_size = 64
        opt.decay = "400"
    # criterion = nn.MSELoss(size_average=False)
    criterion = nn.L1Loss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights["model"].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (opt.lr_decay ** (epoch // opt.step))
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    for iteration, batch in tqdm(enumerate(training_data_loader, 1)):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        output = model(input)
        print(output.size(), target.size())
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(),opt.clip)
        optimizer.step()

        if iteration % 100 == 0:
            print(
                "===> Epoch[{}]({}/{}): Loss: {:.10f}".format(
                    epoch, iteration, len(training_data_loader), loss.data
                ),
                flush=True,
            )

    print(
        "===> Epoch[{}]({}/{}): Loss: {:.10f}".format(
            epoch, iteration, len(training_data_loader), loss.data
        ),
        flush=True,
    )


def save_checkpoint(model, epoch):
    model_out_path = opt.checkpoint + "/model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(opt.checkpoint):
        os.makedirs(opt.checkpoint)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()
