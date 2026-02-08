import os
import warnings
from typing import Callable, Dict, List, Optional

import PIL
import torch
from models.ex_clip import ExCLIP
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as PILImage
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomResizedCrop,
    RandomRotation,
    Resize,
    ToTensor,
)
from tqdm import tqdm

char_size = 150
device = "cuda:0" if torch.cuda.is_available() else "cpu"

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def my_convert_to_rgb(image: PIL.Image) -> PIL.Image:
    """
    Convert gray scale image to rgb image considering alpha channel.

    Parameters
    ----------
    image : PIL.Image
        Gray scale image.

    Returns
    -------
    rgb_image : PIL.Image
        RGB image.
    """

    if image.mode == "RGB":
        return image
    rgb_image = Image.new("RGB", image.size)
    for x in range(rgb_image.width):
        for y in range(rgb_image.height):
            if image.mode == "RGBA":
                gray_value, _, _, alpha = image.getpixel((x, y))
                rgb_value = 255 - int((255 - gray_value) * alpha / 255)
            elif image.mode == "L" or image.mode == "LA":
                gray_value, alpha = image.getpixel((x, y))
                rgb_value = 255 - int((255 - gray_value) * alpha / 255)
            else:
                raise Exception("image mode not supported")

            rgb_image.putpixel((x, y), (rgb_value, rgb_value, rgb_value))
    return rgb_image


def transform_for_aug(
    n_px: int,
    lower_bound_of_scale: float,
    upper_bound_of_scale: float,
    do_color_jitter: bool = False,
) -> list:
    """
    Return the transformation for the augmentation.

    Parameters
    ----------
    n_px : int

    Returns
    -------
    my_compose : Compose
    """
    processes = [
        RandomRotation(180, fill=255, interpolation=BICUBIC),
        RandomResizedCrop(
            n_px,
            scale=(lower_bound_of_scale, upper_bound_of_scale),
            ratio=(1.0, 1.0),
            interpolation=BICUBIC,
        ),
        _convert_image_to_rgb,
    ]
    if do_color_jitter:
        brightness = (0.3, 1.0)
        saturation = (0.0, 1.0)
        hue = (-0.5, 0.5)
        processes.append(
            ColorJitter(brightness=brightness, saturation=saturation, hue=hue)
        )
    processes.append(ToTensor())
    return processes


def transform_for_normalize() -> list:
    """
    Return the transformation for the augmentation.

    Returns
    -------
    my_compose : Compose
    """
    return [
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ]


def transform_for_resize(n_px: int) -> list:
    """
    Return the transformation for the augmentation.

    Parameters
    ----------
    n_px : int

    Returns
    -------
    my_compose : Compose
    """
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            ToTensor(),
        ]
    )


def my_transform(
    n_px: int = 224,
    lower_bound_of_scale: float = 0.35,
    upper_bound_of_scale: float = 1.0,
    do_aug=True,
    do_normalize=True,
    do_color_jitter: bool = False,
) -> Compose:
    """
    Return the transformation for the augmentation.

    Parameters
    ----------
    n_px : int

    Returns
    -------
    my_compose : Compose
    """

    assert do_aug or do_normalize, "do_aug or do_normalize must be True"
    processes = []
    if do_aug:
        processes += transform_for_aug(
            n_px,
            lower_bound_of_scale,
            upper_bound_of_scale,
            do_color_jitter=do_color_jitter,
        )
    if do_normalize:
        processes += transform_for_normalize()
    return Compose(processes)


def generate_all_fonts_embedded_images(
    font_paths: str,
    text: str,
    model: ExCLIP,
    preprocess: Callable[[PILImage], torch.Tensor],
    char_size: int = char_size,
    image_file_dir: Optional[str] = None,
    device: str = device,
    aug_num: int = 1,
    crop_h: Optional[int] = None,
    crop_w: Optional[int] = None,
    font_name_to_image_tensor: Optional[List[torch.Tensor]] = None,
):
    """
    Generate embedded images of the input text for all fonts

    Parameters
    ----------
    font_paths : list of str
        List of font paths
    text : str
        Input text
    model : ExCLIP
        CLIP model
    preprocess : function
        Preprocess function
    char_size : int
        Character size
    image_file_dir : str
        Directory of font images
    device : str
        Device

    Returns
    -------
    result : dict of torch.Tensor
    """
    model.eval()
    with torch.no_grad():
        if font_name_to_image_tensor is None:
            font_name_to_image_tensor = generate_images_for_fonts(
                font_paths=font_paths,
                text=text,
                preprocess=preprocess,
                char_size=char_size,
                image_file_dir=image_file_dir,
                aug_num=aug_num,
                crop_h=crop_h,
                crop_w=crop_w,
            )

        assert font_name_to_image_tensor is not None
        result = {}
        for font_path in font_paths:
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            if font_name not in font_name_to_image_tensor:
                warnings.warn(f"{font_name} is not in font_name_to_image_tensor")
                continue
            image_tensor = font_name_to_image_tensor[font_name].to(device)
            embedded_image = model.encode_image(image_tensor).cpu()
            if embedded_image.shape[0] > 1:
                embedded_image = torch.mean(embedded_image, dim=0).unsqueeze(0).cpu()
            result[font_name] = embedded_image
    return result


def generate_images_for_fonts(
    font_paths: str,
    text: str,
    preprocess: Callable[[PILImage], torch.Tensor],
    char_size: int = char_size,
    image_file_dir: Optional[str] = None,
    aug_num: int = 1,
    crop_h: Optional[int] = None,
    crop_w: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generate tensors of the input text or image file for all fonts

    Parameters
    ----------
    font_paths : list of str
        List of font paths
    text : str
        Input text e.g. "The quick\nbrown fox\njumps over\nthe lazy dog"
    preprocess : function
        Preprocess function
    char_size : int
        Character size in the rendered image
    image_file_dir : str
        Directory of font images
        if image_file_dir is not None, text is ignored
    aug_num : int
        Number of augmentation
    crop_h : int
        Height of the crop
    crop_w : int
        Width of the crop

    Returns
    -------
    result : dict of torch.Tensor
    """
    result = {}
    use_tqdm = False
    if aug_num > 1:
        use_tqdm = True
    font_paths = tqdm(font_paths) if use_tqdm else font_paths
    for font_path in font_paths:
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        image = None
        if image_file_dir is not None:
            image_file_path = os.path.join(image_file_dir, font_name + ".png")
            if not os.path.exists(image_file_path):
                warnings.warn(f"{image_file_path} does not exist")
                continue
            # assert os.path.exists(image_file_path), f"{image_file_path} does not exist"
            image = Image.open(image_file_path)
            image = my_convert_to_rgb(image)
            if crop_h is not None and crop_w is not None:
                w, h = image.size
                image = image.crop(
                    (
                        crop_w,
                        crop_h,
                        w - crop_w,
                        h - crop_h,
                    )
                )
        else:
            line_num = text.count("\n") + 1
            width = int(char_size * len(text) * 1.8 / line_num)
            height = int(char_size * 1.5) * line_num
            font = ImageFont.truetype(font_path, size=char_size)
            image = draw_text_with_new_lines(text, font, width, height)
        assert image is not None
        if aug_num == 1:
            image_tensor = preprocess(image).unsqueeze(0)
        else:
            image_tensor = torch.stack([preprocess(image) for _ in range(aug_num)])
        result[font_name] = image_tensor
    return result


def draw_text_with_new_lines(
    text: str, font: ImageFont.FreeTypeFont, img_width: int, img_height: int
) -> PIL.Image:
    """
    Draw text with new lines

    Parameters
    ----------
    text : str
    font : ImageFont.FreeTypeFont
    img_width : int
    img_height : int

    Returns
    -------
    image : PIL.Image
    """
    image = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    lines = text.split("\n")
    y_text = None
    for line in lines:
        left, top, right, bottom = font.getbbox(line)
        line_height = bottom - top
        line_width = right - left

        if y_text is None:
            y_text = (
                (img_height - line_height * len(lines)) / 2
                if (img_height - line_height * len(lines)) / 2 > 0
                else 0
            )
        x_text = (img_width - line_width) / 2 if (img_width - line_width) / 2 > 0 else 0
        draw.text((x_text - left, y_text - top), line, font=font, fill=(0, 0, 0))
        y_text += line_height
    return image
