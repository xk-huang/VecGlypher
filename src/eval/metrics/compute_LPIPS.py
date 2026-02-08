"""
Download LPIPS model from storage_cli

storage_cli get workspace/hf_downloads/eval_ckpts/vgg16-397923af.pth $HOME/.cache/torch/hub/checkpoints/vgg16-397923af.pth
"""

import lpips
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_metric import BaseMetric


class LPIPSDistanceCalculator(BaseMetric):
    def __init__(self, config=None, device="cuda"):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.config = config

        # NOTE: the model path here is for the five affine transformations.
        # The weight file in the source code directory (it is very small).
        self.model = lpips.LPIPS(net="vgg").to(device)
        self.model.eval()

        self.metric = self.LPIPS

        self.device = device

    def LPIPS(self, tensor_image1, tensor_image2):
        tensor_image1, tensor_image2 = tensor_image1.to(self.device), tensor_image2.to(
            self.device
        )
        return self.model(tensor_image1, tensor_image2)

    def to_tensor_transform(self, img_path):
        if isinstance(img_path, Image.Image):
            img = np.array(img_path)
        else:
            img = lpips.load_image(str(img_path))

        return lpips.im2tensor(img)

    def collate_fn(self, batch):
        gt_imgs, gen_imgs = zip(*batch)
        tensor_gt_imgs = torch.concat(
            [self.to_tensor_transform(img) for img in gt_imgs], dim=0
        )
        tensor_gen_imgs = torch.concat(
            [self.to_tensor_transform(img) for img in gen_imgs], dim=0
        )
        return tensor_gt_imgs, tensor_gen_imgs

    def calculate_score(self, batch, batch_size=128, num_workers=16, update=True):
        gt_images = batch["gt_im"]
        gen_images = batch["gen_im"]

        # Create DataLoader with custom collate function
        data_loader = DataLoader(
            list(zip(gt_images, gen_images)),
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=num_workers,
        )

        values = []
        for tensor_gt_batch, tensor_gen_batch in tqdm(data_loader):
            # Compute LPIPS
            with torch.no_grad():
                lpips_values = self.LPIPS(tensor_gt_batch, tensor_gen_batch)
            # NOTE: (B, 1, 1, 1) -> (B,)
            value = lpips_values.squeeze()
            if value.ndim == 0:
                value = value.unsqueeze(0)
            values.append(value)

        values = torch.cat(values).cpu().detach().tolist()

        if not values:
            print("No valid values found for metric calculation.")
            return float("nan")

        avg_score = sum(values) / len(values)
        if update:
            self.meter.update(avg_score, len(values))
            return self.meter.avg, values
        else:
            return avg_score, values
