import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from numpy import isin
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.functional.multimodal.clip_score import _clip_score_update
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms import ToTensor
from tqdm import tqdm

from .base_metric import BaseMetric

# ref: https://github.com/Lightning-AI/torchmetrics/blob/10c0b84861cf188edc8d1c547364890059e39636/src/torchmetrics/functional/multimodal/clip_score.py#L151
# ref: https://github.com/Lightning-AI/torchmetrics/blob/10c0b84861cf188edc8d1c547364890059e39636/src/torchmetrics/multimodal/clip_score.py#L217


class CLIPScoreCalculator(BaseMetric):
    def __init__(self):
        super().__init__()
        self.class_name = self.__class__.__name__

        model_name_or_path = "openai/clip-vit-base-patch32"
        if os.environ.get("HF_CLIP_MODEL_PATH", None) is not None:
            model_name_or_path = os.environ["HF_CLIP_MODEL_PATH"]
            print(f"Using model path: {model_name_or_path} for CLIPScore")

        self.clip_score = CLIPScore(model_name_or_path=model_name_or_path)
        self.clip_score.to("cuda")
        self.clip_score.eval()

    # def CLIP_Score(self, images, captions):
    #     all_scores = _clip_score_update(
    #         images, captions, self.clip_score.model, self.clip_score.processor
    #     )
    #     return all_scores

    def collate_fn(self, batch):
        gen_imgs, captions = zip(*batch)
        model = self.clip_score.model
        processor = self.clip_score.processor
        # NOTE: transformers/models/clip/image_processing_clip.py:fetch_images requires list not tuple
        gen_imgs = list(gen_imgs)
        img_processed = processor(images=gen_imgs, return_tensors="pt", padding=True)
        text_processed = processor(text=captions, return_tensors="pt", padding=True)
        truncated = False
        if hasattr(model.config, "text_config") and hasattr(
            model.config.text_config, "max_position_embeddings"
        ):
            max_position_embeddings = model.config.text_config.max_position_embeddings
            if text_processed["attention_mask"].shape[-1] > max_position_embeddings:
                current_length = text_processed["attention_mask"].shape[-1]
                print(
                    f"Encountered caption ({current_length}) longer than {max_position_embeddings=}. Will truncate captions to this"
                    " length. If longer captions are needed, initialize argument `model_name_or_path` with a model that"
                    "supports longer sequences.",
                    UserWarning,
                )
                text_processed["attention_mask"] = text_processed["attention_mask"][
                    ..., :max_position_embeddings
                ]
                text_processed["input_ids"] = text_processed["input_ids"][
                    ..., :max_position_embeddings
                ]
                truncated = True
        return (
            img_processed,
            text_processed,
            truncated,
            {"gen_imgs": gen_imgs, "captions": captions},
        )

    def calculate_score(self, batch, batch_size=256, num_workers=16, update=True):
        gen_images = batch["gen_im"]
        captions = batch["caption"]

        def gen_images_to_pil(gen_images):
            for img in gen_images:
                if not isinstance(img, Image.Image):
                    img = Image.open(img)
                yield img

        # Create DataLoader with custom collate function
        data_loader = DataLoader(
            list(zip(gen_images_to_pil(gen_images), captions)),
            collate_fn=self.collate_fn,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        all_scores = []
        model = self.clip_score.model
        for batch_eval in tqdm(data_loader):
            images, captions, truncated, raw = batch_eval
            if truncated:
                raise ValueError("Truncated captions, check your data")
            with torch.no_grad():
                source_features = model.get_image_features(
                    images["pixel_values"].to("cuda")
                )
                target_features = model.get_text_features(
                    captions["input_ids"].to("cuda"),
                    captions["attention_mask"].to("cuda"),
                )
                source_features = source_features / source_features.norm(
                    p=2, dim=-1, keepdim=True
                )
                target_features = target_features / target_features.norm(
                    p=2, dim=-1, keepdim=True
                )
                score = 100 * (source_features * target_features).sum(axis=-1)
            all_scores.append(score)
        all_scores = torch.cat(all_scores, dim=0).cpu().tolist()

        if not all_scores:
            print("No valid scores found for metric calculation.")
            return float("nan"), []

        avg_score = sum(all_scores) / len(all_scores)
        if update:
            self.meter.update(avg_score, len(all_scores))
            return self.meter.avg, all_scores
        else:
            return avg_score, all_scores


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    # Rest of your code...
