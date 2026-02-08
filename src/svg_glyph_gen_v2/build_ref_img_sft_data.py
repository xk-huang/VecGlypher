"""

input_dir=data/processed/filtered_sft/250903-alphanumeric/train_font_family
decoded_input_dir=misc/decoded
img_base64_dir=misc/decoded-img_base64
output_dir=misc/250903-alphanumeric-ref_img/train_font_family

python -m src.serve.decode_to_svg "${input_dir}" "${decoded_input_dir}"
python -m src.eval.svg2img_dir "${decoded_input_dir}" "${img_base64_dir}" --field output --width 192 --height 192

# debug
python -m src.svg_glyph_gen_v2.build_ref_img_sft_data \
    --input_infer_jsonl_dir "${input_dir}" \
    --input_decoded_infer_jsonl_dir "${decoded_input_dir}" \
    --input_infer_img_base64_dir "${img_base64_dir}" \
    --output_media_dir misc/output_media_dir \
    --no_check_img_match \
    --max_samples 10 \

python -m src.svg_glyph_gen_v2.build_ref_img_sft_data \
    --input_infer_jsonl_dir "${input_dir}" \
    --input_decoded_infer_jsonl_dir "${decoded_input_dir}" \
    --input_infer_img_base64_dir "${img_base64_dir}" \
    --output_dir "${output_dir}"

python src/svg_glyph_gen_v2/build_dataset_info.py --add_images "$(dirname ${output_dir})"
python src/svg_glyph_gen_v2/stat_sft_data.py "$(dirname ${output_dir})"

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
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    eval_data_builder = RefImgSFTDataBuilder(args)
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


class RefImgSFTDataBuilder:
    def __init__(self, args):
        self.args = args

        input_decoded_infer_jsonl_dir = Path(args.input_decoded_infer_jsonl_dir)
        self.input_decoded_infer_jsonl_dir = input_decoded_infer_jsonl_dir
        self.input_infer_img_base64_dir = args.input_infer_img_base64_dir
        self.input_infer_jsonl_dir = args.input_infer_jsonl_dir
        output_dataset_name = args.output_dataset_name

        output_dir = args.output_dir
        if output_dir is None:
            output_dir = input_decoded_infer_jsonl_dir.parent / "ref_img_sft_data"
        output_dir = Path(output_dir)
        output_dataset_dir = output_dir / output_dataset_name
        # prepare output dir and logger
        should_skip, logger = prepare_output_dir_and_logger(
            output_dir=output_dataset_dir,
            overwrite=args.overwrite,
            output_log_dir=output_dir / "logs" / output_dataset_name,
        )
        if should_skip:
            exit()

        output_media_dir = args.output_media_dir
        if output_media_dir is None:
            output_media_dir = output_dir / "media"
        output_media_dir = Path(output_media_dir)
        output_dataset_media_dir = output_media_dir / output_dataset_name
        output_dataset_media_dir.mkdir(parents=True, exist_ok=True)
        rel_output_dataset_media_dir = output_dataset_media_dir.relative_to(output_dir)

        self.output_dataset_dir = output_dataset_dir
        self.output_dataset_media_dir = output_dataset_media_dir
        self.rel_output_dataset_media_dir = rel_output_dataset_media_dir
        self.logger = logger
        self.rng = np.random.default_rng(seed=args.seed)
        self.random_pad_ratio = args.random_pad_ratio
        self.max_samples = args.max_samples
        self.check_img_match = args.check_img_match
        self.min_num_ref_imgs = args.min_num_ref_imgs
        self.max_num_ref_imgs = args.max_num_ref_imgs
        logger.info(f"args: {args}")

    def run(self):
        input_decoded_infer_jsonl_dir = self.input_decoded_infer_jsonl_dir
        input_infer_img_base64_dir = self.input_infer_img_base64_dir
        input_infer_jsonl_dir = self.input_infer_jsonl_dir

        # Check data row and image order matching
        num_samples_svg = count_lines(input_decoded_infer_jsonl_dir)
        num_samples_img_base64 = count_lines(input_infer_img_base64_dir)
        if num_samples_svg != num_samples_img_base64:
            error_msg = f"num_samples_svg {num_samples_svg} != num_samples_img_base64 {num_samples_img_base64}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if self.check_img_match:
            self.check_data_img_order_match(
                input_decoded_infer_jsonl_dir,
                input_infer_img_base64_dir,
                num_samples_svg,
            )

        # test single data
        self.test_load_one_sample(
            input_decoded_infer_jsonl_dir, input_infer_img_base64_dir
        )

        # build font family index mapping
        font_glyph_mapping_path = (
            self.output_dataset_media_dir / "font_glyph_mapping.json"
        )
        if font_glyph_mapping_path.exists():
            with open(font_glyph_mapping_path, "r") as f:
                font_glyph_mapping = json.load(f)
            self.logger.info(
                f"font_glyph_mapping loaded from: {font_glyph_mapping_path}"
            )
        else:
            font_glyph_mapping = self.build_font_identifier2glyph_dict(
                input_decoded_infer_jsonl_dir, input_infer_img_base64_dir
            )
            with open(font_glyph_mapping_path, "w") as f:
                json.dump(font_glyph_mapping, f, indent=4)
            self.logger.info(f"font_glyph_mapping saved to: {font_glyph_mapping_path}")

        num_samples_per_font = {k: len(v) for k, v in font_glyph_mapping.items()}
        num_samples_set = set(num_samples_per_font.values())
        self.logger.info(f"num_samples_set: {num_samples_set}")

        # save imgs to media dir
        self.save_imgs_to_media_path_by_img_idx(input_infer_img_base64_dir)

        result_list = []
        for idx, infer_data in enumerate(
            tqdm.tqdm(
                load_jsonl_by_generator(input_infer_jsonl_dir),
                total=num_samples_svg,
                desc="build alpaca data",
            )
        ):
            if self.max_samples is not None and idx >= self.max_samples:
                break
            alpaca_data = self.build_alpaca_data(infer_data, font_glyph_mapping)
            if alpaca_data is None:
                continue
            result_list.append(alpaca_data)
        output_jsonl_path = self.output_dataset_dir / "chunk.jsonl"
        write_jsonl(result_list, output_jsonl_path, logger=self.logger)

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
        selected_media_paths = [
            str(self.rel_output_dataset_media_dir / f"{i}.png")
            for i in selected_media_paths
        ]

        image_placeholder = "<image>" * num_ref_imgs

        instruction_str = "\n".join([STYLE_TEMPLATE, formatted_content_str])
        instruction_str += image_placeholder
        output = infer_data["output"]

        sft_row = {
            "instruction": instruction_str,
            "system": SYSTEM_PROMPT,
            "output": output,
            "images": selected_media_paths,
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

    def build_font_identifier2glyph_dict(
        self, input_decoded_infer_jsonl_dir, input_infer_img_base64_dir
    ):
        """
        Build a dictionary mapping font identifier to a list of glyph data.
        Data format:
        {
            "identifier": {
                "content_str": "A",
                "svg_path_str": "<path d="m654.0,874.0...",
                "img_idx": 0,
            },
            ...
        }
        """
        infer_data_iter = load_jsonl_by_generator(input_decoded_infer_jsonl_dir)
        img_base64_data_iter = load_jsonl_by_generator(input_infer_img_base64_dir)

        font_glyph_mapping = defaultdict(list)
        for img_idx, (infer_data, img_base64_data) in enumerate(
            tqdm.tqdm(
                zip(infer_data_iter, img_base64_data_iter),
                total=count_lines(input_decoded_infer_jsonl_dir),
                desc="build font_identifier2glyph_dict",
            )
        ):
            if self.max_samples is not None and img_idx >= self.max_samples:
                break

            infer_data: Dict[str, str]
            img_base64_data: Dict[str, str]

            img_hash = img_base64_data.get("img_hash", None)
            if img_hash is None:
                raise ValueError(f"img_hash is None: {img_base64_data}")
            metadata = infer_data.get("metadata", None)

            if metadata is None:
                error_msg = f"metadata is None: {infer_data}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            metadata = json.loads(metadata)
            identifier = metadata["identifier"]
            content_str = metadata["content_str"]
            svg_path_str = metadata["sft_output"]

            # NOTE: only keep alphanumeric a-zA-Z0-9
            if not content_str.isalnum():
                continue

            glyph_dict = {
                "content_str": content_str,
                "svg_path_str": svg_path_str,
                "img_idx": img_idx,
            }
            font_glyph_mapping[identifier].append(glyph_dict)
        return font_glyph_mapping

    def test_load_one_sample(
        self, input_decoded_infer_jsonl_dir, input_infer_img_base64_dir
    ):
        infer_data = next(iter(load_jsonl_by_generator(input_decoded_infer_jsonl_dir)))
        img_base64_data = next(
            iter(load_jsonl_by_generator(input_infer_img_base64_dir))
        )

        img_base64_data: Dict[str, str]
        img_hash = img_base64_data.get("img_hash", None)
        img_base64 = img_base64_data.get("img_base64", None)
        if img_hash is None:
            error_msg = f"img_hash is None: {img_base64_data}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if img_base64 is None:
            error_msg = f"img_base64 is None: {img_base64_data}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        image_data = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(image_data))
        image.load()

        # randomly pad the image
        width, height = image.size
        width_max_pad = int(self.random_pad_ratio * width / 2)
        height_max_pad = int(self.random_pad_ratio * height / 2)
        random_padded_image = self.random_pad(image, width_max_pad, height_max_pad)

    def save_imgs_to_media_path_by_img_idx(self, input_infer_img_base64_dir):
        for img_idx, img_base64_data in enumerate(
            tqdm.tqdm(
                load_jsonl_by_generator(input_infer_img_base64_dir),
                total=count_lines(input_infer_img_base64_dir),
                desc="save_imgs_to_media_dir",
            )
        ):
            if self.max_samples is not None and img_idx >= self.max_samples:
                break
            img_base64_data: Dict[str, str]

            img_hash = img_base64_data.get("img_hash", None)
            img_base64 = img_base64_data.get("img_base64", None)
            if img_hash is None:
                error_msg = f"img_hash is None: {img_base64_data}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            if img_base64 is None:
                error_msg = f"img_base64 is None: {img_base64_data}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            output_media_path = self.output_dataset_media_dir / f"{img_idx}.png"
            if output_media_path.exists():
                raise ValueError(
                    f"output_media_path {output_media_path} already exists, manually delete it first."
                )

            image_data = base64.b64decode(img_base64)
            image = Image.open(io.BytesIO(image_data))
            image.load()

            # randomly pad the image
            width, height = image.size
            width_max_pad = int(self.random_pad_ratio * width / 2)
            height_max_pad = int(self.random_pad_ratio * height / 2)
            random_padded_image = self.random_pad(image, width_max_pad, height_max_pad)

            random_padded_image.save(
                output_media_path, format="PNG", optimize=False, compress_level=1
            )
        self.logger.info(f"Saved imgs to: {self.output_dataset_media_dir}")

    def check_data_img_order_match(
        self, input_decoded_infer_jsonl_dir, input_infer_img_base64_dir, num_samples_svg
    ):
        infer_data_iter = load_jsonl_by_generator(input_decoded_infer_jsonl_dir)
        img_base64_data_iter = load_jsonl_by_generator(input_infer_img_base64_dir)

        for infer_data, img_base64_data in tqdm.tqdm(
            zip(infer_data_iter, img_base64_data_iter),
            total=num_samples_svg,
            desc="build font family index mapping",
        ):
            infer_data: Dict[str, str]
            img_base64_data: Dict[str, str]
            svg_text = infer_data.get(self.args.field, None)
            if svg_text is None:
                raise ValueError(f"svg_text is None: {infer_data}")
            img_hash = blake2_hash(svg_text)
            img_hash_from_img_base64 = img_base64_data["img_hash"]

            if img_hash != img_hash_from_img_base64:
                error_msg = f"img_hash {img_hash} != img_hash_from_img_base64 {img_hash_from_img_base64}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        self.logger.info(f"Data and image order all matched.")

    def random_pad(
        self, img, width_max_pad=20, height_max_pad=20, fill_color=(255, 255, 255)
    ):
        """
        Randomly pad an image by up to `max_pad` pixels on each side.

        Args:
            img (PIL.Image): The input image.
            max_pad (int): Maximum padding in pixels (each side).
            fill_color (tuple/int): Padding color (RGB or grayscale).

        Returns:
            PIL.Image: The padded image.
        """

        left = int(self.rng.integers(0, width_max_pad))
        right = int(self.rng.integers(0, width_max_pad))
        top = int(self.rng.integers(0, height_max_pad))
        bottom = int(self.rng.integers(0, height_max_pad))

        padded_img = ImageOps.expand(
            img, border=(left, top, right, bottom), fill=fill_color
        )
        return padded_img


if __name__ == "__main__":
    main()
