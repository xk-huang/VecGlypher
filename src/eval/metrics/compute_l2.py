import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm

from .base_metric import BaseMetric


class L2DistanceCalculator(BaseMetric):
    def __init__(self, config=None, masked_l2=False):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.config = config
        # self.metric = self.l2_distance
        # self.masked_l2 = masked_l2
        self.to_tensor = ToTensor()

    # def l2_distance(self, **kwargs):
    #     image1 = kwargs.get("gt_im")
    #     image2 = kwargs.get("gen_im")
    #     image1_tensor = ToTensor()(image1)
    #     image2_tensor = ToTensor()(image2)

    #     if self.masked_l2:
    #         # Create binary masks: 0 for white pixels, 1 for non-white pixels
    #         mask1 = (image1_tensor != 1).any(dim=0).float()
    #         mask2 = (image2_tensor != 1).any(dim=0).float()

    #         # Create a combined mask for overlapping non-white pixels
    #         combined_mask = mask1 * mask2

    #         # Apply the combined mask to both images
    #         image1_tensor = image1_tensor * combined_mask.unsqueeze(0)
    #         image2_tensor = image2_tensor * combined_mask.unsqueeze(0)

    #     # Compute mean squared error
    #     mse = F.mse_loss(image1_tensor, image2_tensor)
    #     return mse.item()

    def to_tensor_transform(self, pil_img):
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.open(pil_img)
        return self.to_tensor(pil_img)

    def collate_fn(self, batch):
        gt_imgs, gen_imgs = zip(*batch)
        tensor_gt_imgs = torch.stack([self.to_tensor_transform(img) for img in gt_imgs])
        tensor_gen_imgs = torch.stack(
            [self.to_tensor_transform(img) for img in gen_imgs]
        )
        return tensor_gt_imgs, tensor_gen_imgs

    def calculate_score(self, batch, batch_size=256, num_workers=16, update=True):
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
            tensor_gt_batch = tensor_gt_batch.to("cuda")
            tensor_gen_batch = tensor_gen_batch.to("cuda")
            with torch.no_grad():
                l2_values = F.mse_loss(
                    tensor_gt_batch, tensor_gen_batch, reduction="none"
                )
                l2_values = l2_values.mean(dim=[1, 2, 3])
            values.append(l2_values)
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
