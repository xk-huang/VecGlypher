"""
Copied from star-vector, but the code is largely modified to support batch processing
"""

import os

import numpy as np

from tqdm import tqdm

from .compute_clip_score import CLIPScoreCalculator
from .compute_dino_score import DINOScoreCalculator
from .compute_fid import FIDCalculator
from .compute_img_proc_sim import ImageProcessingDistanceCalculator

from .compute_l2 import L2DistanceCalculator
from .compute_LPIPS import LPIPSDistanceCalculator
from .count_token_length import CountTokenLength
from .util import AverageMeter


class SVGMetrics:
    def __init__(self, config=None):
        self.class_name = self.__class__.__name__

        default_config = {
            "L2": True,
            # "Masked-L2": False,
            "PSNR": True,
            "PSNR-auto": True,
            "SSIM": True,
            "SSIM-auto": True,
            # "MS-SSIM": False, # NOTE: only 128x128, no need for using multi-scale
            "LPIPS": True,
            "FID": True,
            "FID_clip": True,
            "CLIPScore": True,
            "DinoScore": True,
            "CountTokenLength": False,
            "ratio_post_processed": False,
            "ratio_non_compiling": False,
        }
        self.config = config or default_config

        self.metrics = {
            "L2": L2DistanceCalculator,
            "Masked-L2": lambda: L2DistanceCalculator(masked_l2=True),
            "PSNR": lambda: ImageProcessingDistanceCalculator("psnr"),
            "PSNR-auto": lambda: ImageProcessingDistanceCalculator("psnr", "auto"),
            "SSIM": lambda: ImageProcessingDistanceCalculator("ssim"),
            "SSIM-auto": lambda: ImageProcessingDistanceCalculator("ssim", "auto"),
            "MS-SSIM": lambda: ImageProcessingDistanceCalculator("ms-ssim"),
            "LPIPS": LPIPSDistanceCalculator,
            "FID": lambda: FIDCalculator(model_name="InceptionV3"),
            "FID_clip": lambda: FIDCalculator(model_name="ViT-B/32"),
            "CLIPScore": CLIPScoreCalculator,
            "CountTokenLength": CountTokenLength,
            "ratio_post_processed": AverageMeter,
            "ratio_non_compiling": AverageMeter,
            "DinoScore": DINOScoreCalculator,
        }

        self.active_metrics = {
            k: v() for k, v in self.metrics.items() if self.config.get(k)
        }

    def reset(self):
        for metric in self.active_metrics.values():
            metric.reset()

    def batch_contains_raster(self, batch):
        return "gt_im" in batch and "gen_im" in batch

    def batch_contains_svg(self, batch):
        return "gt_svg" in batch and "gen_svg" in batch

    def calculate_metrics(self, batch, update=True):

        avg_results_dict = {}
        all_results_dict = {}

        # initialize all_results_dict
        for i, json_item in enumerate(batch["json"]):
            sample_id = self.get_sample_id(json_item)
            if sample_id is None:
                raise ValueError(
                    f"Could not find 'outpath_filename' or 'sample_id' in batch['json'][{i}]"
                )
            all_results_dict[sample_id] = {}

        for metric_name, metric in self.active_metrics.items():
            print(f"Calculating {metric_name}...")

            # Handle metrics that return both average and per-sample results
            if metric_name in [
                "L2",
                "Masked-L2",
                "PSNR",
                "SSIM",
                "PSNR-auto",
                "SSIM-auto",
                "MS-SSIM",
                "CLIPScore",
                "LPIPS",
                "CountTokenLength",
                "DinoScore",
            ]:
                avg_result, list_result = metric.calculate_score(batch, update=update)
                avg_results_dict[metric_name] = avg_result

                # Store individual results
                for i, result in enumerate(list_result):
                    sample_id = self.get_sample_id(batch["json"][i])
                    all_results_dict[sample_id][metric_name] = result

            # Handle FID metrics that only return average
            elif metric_name in ["FID", "FID_clip"]:
                avg_results_dict[metric_name] = metric.calculate_score(batch)

            # Handle other metrics (ratio metrics)
            else:
                self._handle_ratio_metric(
                    metric_name, metric, batch, avg_results_dict, all_results_dict
                )

            metric.reset()

        # NOTE: post process add mse-to-psnr
        if "L2" in avg_results_dict:
            avg_results_dict["PSNR_from_avg_L2"] = self._psnr_from_mse(
                avg_results_dict["L2"], data_range=1.0
            )

        print("Average results: \n", avg_results_dict)
        return avg_results_dict, all_results_dict

    # def calculate_fid(self, batch):
    #     return self.active_metrics["FID"].calculate_score(batch).item()

    def get_average_metrics(self):
        metrics = {}
        for metric_name, metric in self.active_metrics.items():
            if hasattr(metric, "avg"):
                metrics[metric_name] = metric.avg
            elif hasattr(metric, "get_average_score"):
                metrics[metric_name] = metric.get_average_score()
        return metrics

    def _handle_ratio_metric(
        self, metric_name, metric, batch, avg_results_dict, all_results_dict
    ):
        """Helper method to handle ratio-based metrics."""
        metric_key = metric_name.replace("avg_", "").replace("ratio_", "")

        for item in batch["json"]:
            sample_id = self.get_sample_id(item)
            value = item[metric_key]
            all_results_dict[sample_id][metric_name] = value
            metric.update(value, 1)

        avg_results_dict[metric_name] = metric.avg

    @staticmethod
    def get_sample_id(json_item):
        return json_item.get("outpath_filename") or json_item.get("sample_id")

    @staticmethod
    def _psnr_from_mse(mse, data_range=1.0):
        if mse == 0:
            return float("inf")  # perfect match
        return 10 * np.log10((data_range**2) / mse)
