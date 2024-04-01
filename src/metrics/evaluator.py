import torch
import numpy as np

from src.utils import tensor2img
from src.metrics import calculate_psnr, calculate_ssim

from typing import Dict


class Evaluator:
    def __init__(self):
        pass

    def eval(self, seq0: torch.Tensor, seq1: torch.Tensor) -> Dict[str, np.ndarray]:
        """Evaluates similarity/dissimilarity between two sequences.

        Args:
            seq0: First sequence of shape in (T,C,H,W).
            seq1: Second sequence of shape in (T,C,H,W).

        Returns:
            A dictionary containing evaluation results.
            Keys: Name of metric.
            Values: 1d array containing evaluation result of each image.
        """
        list_psnr = []
        list_ssim = []

        for img0, img1 in zip(seq0, seq1):
            img0 = tensor2img(img0)
            img1 = tensor2img(img1)

            psnr = calculate_psnr(
                img0, img1, crop_border=0, input_order="HWC", test_y_channel=True
            )
            list_psnr.append(psnr)

            ssim = calculate_ssim(
                img0, img1, crop_border=0, input_order="HWC", test_y_channel=True
            )
            list_ssim.append(ssim)

        result = {
            "psnr": np.asarray(list_psnr),
            "ssim": np.asarray(list_ssim)
        }
        return result


def squeeze_eval_result(eval_result: Dict[str, np.ndarray]) -> Dict[str, float]:
    squeezed_result = {}
    for metric in eval_result:
        squeezed_result[metric] = float(eval_result[metric].mean())
    return squeezed_result
