import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from .base_metric import BaseMetric


class DINOScoreCalculator(BaseMetric):
    def __init__(self, config=None, device="cuda"):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.config = config
        model_name = None
        if os.environ.get("HF_DINO_MODEL_PATH", None) is not None:
            model_name = os.environ["HF_DINO_MODEL_PATH"]
        self.model, self.processor = self.get_DINOv2_model("base", model_name)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.cos = nn.CosineSimilarity(dim=1)

        # self.metric = self.calculate_DINOv2_similarity_score

    def get_DINOv2_model(self, model_size, model_name=None):
        if model_name is not None:
            print(f"Loading DINOv2 model from {model_name} for DINOScore")
            return AutoModel.from_pretrained(
                model_name
            ), AutoImageProcessor.from_pretrained(model_name)

        if model_size == "small":
            model_name = "facebook/dinov2-small"
        elif model_size == "base":
            model_name = "facebook/dinov2-base"
        elif model_size == "large":
            model_name = "facebook/dinov2-large"
        else:
            raise ValueError(
                f"model_size should be either 'small', 'base' or 'large', got {model_size}"
            )
        return AutoModel.from_pretrained(
            model_name
        ), AutoImageProcessor.from_pretrained(model_name)

    # def process_input(self, image, processor):
    #     if isinstance(image, str):
    #         image = Image.open(image)
    #     if isinstance(image, Image.Image):
    #         with torch.no_grad():
    #             inputs = processor(images=image, return_tensors="pt").to(self.device)
    #             outputs = self.model(**inputs)
    #             features = outputs.last_hidden_state.mean(dim=1)
    #     elif isinstance(image, torch.Tensor):
    #         features = image.unsqueeze(0) if image.dim() == 1 else image
    #     else:
    #         raise ValueError(
    #             "Input must be a file path, PIL Image, or tensor of features"
    #         )
    #     return features

    # def calculate_DINOv2_similarity_score(self, **kwargs):
    #     image1 = kwargs.get("gt_im")
    #     image2 = kwargs.get("gen_im")
    #     features1 = self.process_input(image1, self.processor)
    #     features2 = self.process_input(image2, self.processor)

    #     cos = nn.CosineSimilarity(dim=1)
    #     sim = cos(features1, features2).item()
    #     sim = (sim + 1) / 2

    #     return sim

    def collate_fn(self, batch):
        raw_gt_imgs, raw_gen_imgs = zip(*batch)
        gt_imgs = []
        for gt_img in raw_gt_imgs:
            if not isinstance(gt_img, Image.Image):
                gt_img = Image.open(gt_img)
            gt_imgs.append(gt_img)
        gen_imgs = []
        for gen_img in raw_gen_imgs:
            if not isinstance(gen_img, Image.Image):
                gen_img = Image.open(gen_img)
            gen_imgs.append(gen_img)
        tensor_gt_imgs = self.processor(images=gt_imgs, return_tensors="pt")
        tensor_gen_imgs = self.processor(images=gen_imgs, return_tensors="pt")
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
                gt_out = self.model(**tensor_gt_batch)
                gen_out = self.model(**tensor_gen_batch)

            gt_feature = gt_out.last_hidden_state.mean(dim=1)
            gen_feature = gen_out.last_hidden_state.mean(dim=1)

            value = self.cos(gt_feature, gen_feature)
            value = (value + 1) / 2
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
