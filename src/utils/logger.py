import copy
import os
from typing import Optional, OrderedDict, Dict

import numpy as np
import yaml
from yaml import CDumper as Dumper

from src.metrics import squeeze_eval_result


class Logger:
    def __init__(
        self, log_dir_name: Optional[str] = None, log_file_name: Optional[str] = None
    ):
        if log_dir_name is not None:
            os.makedirs(log_dir_name, exist_ok=True)
        self.log_dir_name = log_dir_name
        self.log_file_path = os.path.join(log_dir_name, log_file_name)

        # Maps the name of category to evaluation result.
        self.base_log: Dict[str, Dict[str, float]] = {}
        self.stu_log: Dict[str, Dict[str, float]] = {}
        self.adapt_log: Dict[str, Dict[str, float]] = {}

    def log(self, message: str) -> None:
        if self.log_file_path is not None:
            with open(self.log_file_path, "a") as f:
                f.write(message + "\n")
        print(message, flush=True)

    def log_base_result(self, solver, eval_result: Dict[str, np.ndarray]) -> None:
        eval_result = squeeze_eval_result(eval_result)
        self.base_log[solver.category] = eval_result

        self.log(
            f"Clip: {solver.category} | Baseline: {solver.base_model.__class__.__name__}\n"
            f'  - PSNR: {eval_result["psnr"]:.2f}\n'
            f'  - SSIM: {eval_result["ssim"]:.4f}\n'
            f'  - LPIPS: {eval_result["lpips"]:.5f}\n'
        )

    def log_stu_result(self, solver, eval_result: Dict[str, np.ndarray]) -> None:
        eval_result = squeeze_eval_result(eval_result)
        self.stu_log[solver.category] = eval_result

        self.log(
            f"Clip: {solver.category} | Student: {solver.stu_model.__class__.__name__}\n"
            f'  - PSNR: {eval_result["psnr"]:.2f}\n'
            f'  - SSIM: {eval_result["ssim"]:.4f}\n'
            f'  - LPIPS: {eval_result["lpips"]:.5f}\n'
        )

    def log_adapt_result(
        self, solver, eval_result: Dict[str, np.ndarray], cur_iter: int
    ) -> None:
        eval_result = squeeze_eval_result(eval_result)
        self.adapt_log[solver.category] = eval_result

        gain = {}
        gain_percent = {}
        for metric in ["psnr", "ssim", "lpips"]:
            gain[metric] = eval_result[metric] - self.base_log[solver.category][metric]
            gain_percent[metric] = (
                gain[metric] / self.base_log[solver.category][metric] * 100
            )

        self.log(
            f"Clip: {solver.category} | Adapted: {cur_iter:d}/{solver.total_iters:d}\n"
            f'  - PSNR: {eval_result["psnr"]:.2f}, gain: {gain["psnr"]:+.2f} ({gain_percent["psnr"]:+.2f}%)\n'
            f'  - SSIM: {eval_result["ssim"]:.4f}, gain: {gain["ssim"]:+.4f} ({gain_percent["ssim"]:+.2f}%)\n'
            f'  - LPIPS: {eval_result["lpips"]:.5f}, gain: {gain["lpips"]:+.5f} ({gain_percent["lpips"]:+.2f}%)\n'
        )

    def log_average(self) -> None:

        avg_base = {}
        avg_adapt = {}
        gain = {}
        gain_percent = {}

        for metric in ["psnr", "ssim", "lpips"]:
            avg_base[metric] = np.average(
                [log[metric] for log in self.base_log.values()]
            )
            avg_adapt[metric] = np.average(
                [log[metric] for log in self.adapt_log.values()]
            )
            gain[metric] = avg_adapt[metric] - avg_base[metric]
            gain_percent[metric] = gain[metric] / avg_base[metric] * 100

        self.log(
            f"\n"
            f"========== Average ==========\n"
            f'  - PSNR: {avg_base["psnr"]:.2f} -> {avg_adapt["psnr"]:.2f}, '
            f'gain: {gain["psnr"]:+.2f} ({gain_percent["psnr"]:+.2f}%)\n'
            f'  - SSIM: {avg_base["ssim"]:.4f} -> {avg_adapt["ssim"]:.4f}, '
            f'gain: {gain["ssim"]:+.4f} ({gain_percent["ssim"]:+.2f}%)\n'
            f'  - SSIM: {avg_base["lpips"]:.5f} -> {avg_adapt["lpips"]:.5f}, '
            f'gain: {gain["lpips"]:+.5f} ({gain_percent["lpips"]:+.2f}%)\n'
        )

    def log_option(
        self, opt: OrderedDict, opt_file_name: Optional[str] = "options.yml"
    ):
        Dumper.add_representer(
            OrderedDict, lambda dumper, data: dumper.represent_dict(data.items())
        )
        opt_file_path = os.path.join(self.log_dir_name, opt_file_name)
        with open(opt_file_path, "w") as f:
            yaml.dump(dict(opt), f, default_flow_style=False, Dumper=Dumper)
