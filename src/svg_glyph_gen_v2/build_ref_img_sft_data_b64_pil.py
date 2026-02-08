"""

input_dir=data/processed/filtered_sft/250903-alphanumeric/ood_font_family_decon
decoded_input_dir=misc/250903-alphanumeric-ref_img/tmp/ood_font_family_decon-decoded
img_base64_dir=misc/250903-alphanumeric-ref_img/tmp/ood_font_family_decon-decoded-img_base64
output_dir=misc/250903-alphanumeric-ref_img-b64_pil/
output_dataset_name=ood_font_family_decon

python -m src.serve.decode_to_svg "${input_dir}" "${decoded_input_dir}"
python -m src.eval.svg2img_dir "${decoded_input_dir}" "${img_base64_dir}" --field output --width 192 --height 192

python -m src.svg_glyph_gen_v2.build_ref_img_sft_data_b64_pil \
    --input_infer_jsonl_dir "${input_dir}" \
    --input_decoded_infer_jsonl_dir "${decoded_input_dir}" \
    --input_infer_img_base64_dir "${img_base64_dir}" \
    --output_dir "${output_dir}" \
    --output_dataset_name "${output_dataset_name}"

Ref:
- src/eval/build_eval_data.py
- src/svg_glyph_gen_v2/build_sft_data_v2.py
- src/eval/svg2img_dir.py
"""

import base64
import io
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import click
import numpy as np
import tqdm
from PIL import Image, ImageOps

from ..svg_glyph_gen_v2.filter_by_pangram_svg import blake2_hash
from ..svg_glyph_gen_v2.utils import (
    count_lines,
    load_jsonl,
    load_jsonl_by_generator,
    prepare_output_dir_and_logger,
    write_jsonl,
)
from .build_ref_img_sft_data import RefImgSFTDataBuilder


@click.command()
@click.option(
    "--input_infer_jsonl_dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    required=True,
)
@click.option(
    "--input_decoded_infer_jsonl_dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    required=True,
)
@click.option(
    "--input_infer_img_base64_dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    required=True,
)
@click.option(
    "--field",
    default="output",
    required=True,
    help="JSON field containing SVG text.",
)
@click.option(
    "--output_dir",
    type=click.Path(),
    default=None,
)
@click.option(
    "--output_dataset_name", default="dataset", help="Name of the output dataset."
)
@click.option("--seed", default=42, type=int)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--random_pad_ratio", default=0.25, type=float)
@click.option("--max_samples", default=None, type=int)
@click.option("--output_media_dir", default=None, type=str)
@click.option("--check_img_match/--no_check_img_match", is_flag=True, default=False)
@click.option("--min_num_ref_imgs", default=1, type=int)
@click.option("--max_num_ref_imgs", default=8, type=int)
@click.option(
    "--add_image_placeholder/--no_add_image_placeholder", is_flag=True, default=False
)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    eval_data_builder = RefImgSFTDataB64PILBuilder(args)
    eval_data_builder.run()


SYSTEM_PROMPT = """You are a specialized vector glyph designer creating SVG path elements.

CRITICAL REQUIREMENTS:
- Each glyph must be a complete, self-contained <path> element, in reading order of the given text.
- Terminate each <path> element with a newline character
- Output ONLY valid SVG <path> elements
"""

STYLE_TEMPLATE = """Font design requirements: faithfully match the provided reference images for style and metrics.
"""

CONTENT_TEMPLATE = """Text content: {content_str}
"""


class RefImgSFTDataB64PILBuilder(RefImgSFTDataBuilder):

    def __init__(self, args):
        super().__init__(args)
        self.add_image_placeholder = args.add_image_placeholder

    def build_alpaca_data(self, infer_data, font_glyph_mapping):
        infer_data: Dict[str, str]

        raw_metadata = infer_data.get("metadata", None)
        if raw_metadata is None:
            error_msg = f"metadata is None: {infer_data}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        metadata = json.loads(raw_metadata)

        content_str = metadata["content_str"]
        # NOTE: only keep alphanumeric a-zA-Z0-9
        if not content_str.isalnum():
            return None
        formatted_content_str = CONTENT_TEMPLATE.format(content_str=content_str)

        identifier = metadata["identifier"]
        if identifier not in font_glyph_mapping:
            error_msg = f"identifier {identifier} not in font_glyph_mapping"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        glyph_mapping = font_glyph_mapping[identifier]

        num_ref_imgs = self.rng.integers(
            self.min_num_ref_imgs, self.max_num_ref_imgs + 1
        )
        # NOTE: int64 is not JSON serializable
        num_ref_imgs = int(num_ref_imgs)
        num_ref_imgs = min(num_ref_imgs, len(glyph_mapping))

        selected_glyph_mapping = self.rng.choice(
            glyph_mapping, num_ref_imgs, replace=False
        )
        selected_media_paths = [i["img_idx"] for i in selected_glyph_mapping]
        # NOTE: pathlib.Path is not JSON serializable
        # NOTE: we need to load the image here, thus we do not use rel_output_dataset_media_dir
        selected_media_paths = [
            str(self.output_dataset_media_dir / f"{i}.png")
            for i in selected_media_paths
        ]
        selected_media = [load_pil_img_to_base64(i) for i in selected_media_paths]

        if self.add_image_placeholder:
            image_placeholder = "<image>" * num_ref_imgs
        else:
            image_placeholder = ""

        instruction_str = "\n".join([STYLE_TEMPLATE, formatted_content_str])
        instruction_str += image_placeholder
        output = infer_data["output"]

        sft_row = {
            "instruction": instruction_str,
            "system": SYSTEM_PROMPT,
            "output": output,
            "images": selected_media,
        }

        # add metadata
        metadata = metadata.copy()
        metadata["num_ref_imgs"] = num_ref_imgs
        ref_svg_path_strs = [i["svg_path_str"] for i in selected_glyph_mapping]
        ref_content_strs = [i["content_str"] for i in selected_glyph_mapping]
        metadata["ref_svg_path_strs"] = ref_svg_path_strs
        metadata["ref_content_strs"] = ref_content_strs
        metadata.update({f"sft_{k}": v for k, v in sft_row.items()})
        sft_row.update({"metadata": json.dumps(metadata)})
        return sft_row


def load_pil_img_to_base64(img_path):
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64


if __name__ == "__main__":
    main()
