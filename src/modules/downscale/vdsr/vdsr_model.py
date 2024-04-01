import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        # self.net = torch.load('model/model_epoch_50.pth')["model"]
        # self.net = torch.load(os.path.join(os.getcwd(), "ssa/modules/vdsr/model/ycbcr/official_pretrained.pth"))["model"]
        # self.net = torch.load(os.path.join(os.getcwd(), "ssa/modules/vdsr/model/ycbcr/naive_50.pth"))["model"]
        self.net = torch.load(
            os.path.join(os.getcwd(), "ssa/modules/vdsr/model/rgb/patch_epoch_23.pth")
        )["model"]
        # self.net = torch.load(os.path.join(os.getcwd(), "ssa/modules/vdsr/checkpoint/1000_100/model_epoch_500.pth"))["model"]
        # for param in self.net.parameters():
        #    param.requires_grad = False

    # @torch.no_grad()
    def forward(self, x):
        """
        ycbcr = rgb_to_ycbcr(x)
        y = torch.unsqueeze(ycbcr[:,0,:,:], 1)
        hy = self.net(y)
        hy = torch.clamp(hy, 0, 1)
        hycbcr = torch.stack(
            [hy[:,0,:,:], ycbcr[:,1,:,:], ycbcr[:,2,:,:]], dim=1)
        x = ycbcr_to_rgb(hycbcr)
        x = torch.clamp(x, 0, 1)
        return x
        """
        return self.net(x)


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


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"
        )

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    r: torch.Tensor = y + 1.403 * cr_shifted
    g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: torch.Tensor = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)


class RgbToYcbcr(nn.Module):
    r"""Convert an image from RGB to YCbCr.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        YCbCr version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = RgbToYcbcr()
        >>> output = ycbcr(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_ycbcr(image)


class YcbcrToRgb(nn.Module):
    r"""Convert an image from YCbCr to Rgb.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        RGB version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YcbcrToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return ycbcr_to_rgb(image)


if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    import torchvision.transforms as transforms

    vdsr = VDSR().cuda()

    img = Image.open("calendar.png")
    img = np.array(img).astype(np.float32) / 255
    print(img.shape, np.min(img), np.mean(img), np.max(img))

    tf = transforms.ToTensor()
    tensor = tf(img)
    tensor = torch.unsqueeze(tensor, 0).cuda()
    print(tensor.size(), torch.min(tensor), torch.mean(tensor), torch.max(tensor))

    tensor = vdsr(tensor)

    print(tensor.size(), torch.min(tensor), torch.mean(tensor), torch.max(tensor))
