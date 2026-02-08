from functools import partial
from os import execlpe

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchvision.transforms import Normalize, ToPILImage, ToTensor
from tqdm import tqdm

from .base_metric import BaseMetric

METRIC_NAME2CLASS = {
    "ssim": StructuralSimilarityIndexMeasure,
    "psnr": partial(PeakSignalNoiseRatio, dim=[1, 2, 3]),
    "ms-ssim": MultiScaleStructuralSimilarityIndexMeasure,
}
# NOTE: psnr in torchmetrics, when dim is None, by default reduce all dims.
# Since only reduction="none" is not enought to determine the dim, we need to specify the dim explicitly.


class ImageProcessingDistanceCalculator(BaseMetric):
    def __init__(self, metric_name, accum="manual", config=None):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.config = config
        self.to_tensor = ToTensor()
        self.metric_name = metric_name
        reduction = None
        if accum == "auto":
            reduction = "elementwise_mean"
        elif accum == "manual":
            reduction = "none"
        else:
            raise ValueError(f"Unknown accum method: {accum}")

        self.metric_func = METRIC_NAME2CLASS[self.metric_name](
            data_range=1.0, reduction=reduction
        ).to("cuda")
        self.to_pil = ToPILImage()
        self.accum = accum

    # def compute_SSIM(self, **kwargs):
    #     image1 = kwargs.get("gt_im")
    #     image2 = kwargs.get("gen_im")
    #     image1 = Image.open(image1)
    #     image2 = Image.open(image2)
    #     win_size = kwargs.get("win_size", 11)  # Increase win_size for more accuracy
    #     channel_axis = kwargs.get("channel_axis", -1)  # Default channel_axis to -1
    #     sigma = kwargs.get("sigma", 1.5)  # Add sigma parameter for Gaussian filter

    #     # Convert images to numpy arrays if they aren't already
    #     img1_np = np.array(image1)
    #     img2_np = np.array(image2)

    #     # Check if images are grayscale or RGB
    #     if len(img1_np.shape) == 3 and img1_np.shape[2] == 3:
    #         # Compute SSIM for RGB images
    #         score, _ = ssim(
    #             img1_np,
    #             img2_np,
    #             win_size=win_size,
    #             channel_axis=channel_axis,
    #             sigma=sigma,
    #             full=True,
    #         )
    #     else:
    #         # Convert to grayscale if not already
    #         if len(img1_np.shape) == 3:
    #             img1_np = np.mean(img1_np, axis=2)
    #             img2_np = np.mean(img2_np, axis=2)

    #         score, _ = ssim(img1_np, img2_np, win_size=win_size, sigma=sigma, full=True)

    #     return score
    def to_tensor_transform(self, pil_img):
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.open(pil_img)
        return self.to_tensor(pil_img)

    def collate_fn(self, batch):
        gt_imgs, gen_imgs, json_dict = zip(*batch)
        tensor_gt_imgs = torch.stack([self.to_tensor_transform(img) for img in gt_imgs])
        tensor_gen_imgs = torch.stack(
            [self.to_tensor_transform(img) for img in gen_imgs]
        )
        return tensor_gt_imgs, tensor_gen_imgs, json_dict

    def calculate_score(self, batch, batch_size=256, num_workers=16, update=True):
        gt_images = batch["gt_im"]
        gen_images = batch["gen_im"]
        json_dict = batch["json"]

        # Create DataLoader with custom collate function
        data_loader = DataLoader(
            list(zip(gt_images, gen_images, json_dict)),
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=num_workers,
        )

        if self.accum == "manual":
            values = []
            for tensor_gt_batch, tensor_gen_batch, json_dict in tqdm(data_loader):
                tensor_gt_batch = tensor_gt_batch.to("cuda")
                tensor_gen_batch = tensor_gen_batch.to("cuda")
                with torch.no_grad():
                    value = self.metric_func(tensor_gt_batch, tensor_gen_batch)

                # NOTE: if any entry is inf, skip the batch,
                # remove those inf entries
                if torch.any(torch.isinf(value)):
                    value = value[~torch.isinf(value)]
                # NOTE: if it is a scalar, unsqueeze to make it a 1D tensor
                # Otherwise, concat is not working
                if value.ndim == 0:
                    value = value.unsqueeze(0)
                values.append(value)

            values = torch.cat(values).cpu().detach().tolist()
            if not values:
                print("No valid values found for metric calculation.")
                return float("nan")

            avg_score = sum(values) / len(values)
        elif self.accum == "auto":
            for tensor_gt_batch, tensor_gen_batch, json_dict in tqdm(data_loader):
                tensor_gt_batch = tensor_gt_batch.to("cuda")
                tensor_gen_batch = tensor_gen_batch.to("cuda")
                with torch.no_grad():
                    self.metric_func.update(tensor_gt_batch, tensor_gen_batch)
            with torch.no_grad():
                avg_score = self.metric_func.compute()
                avg_score = avg_score.item()

            # NOTE: pseudo value for each entry, since we don't have the actual values
            values = [avg_score] * len(json_dict)
        else:
            raise ValueError(f"Unknown accum method: {self.accum}")

        if update:
            self.meter.update(avg_score, len(batch))
            return self.meter.avg, values
        else:
            return avg_score, values
