"""
python -m src.eval.build_eval_data \
    --input_infer_jsonl_dir [INPUT_INFER_JSONL_DIR]
    --input_infer_img_base64_dir [INPUT_INFER_IMG_BASE64_DIR]
    --field [FIELD]

Run infer with vllm, refer to: `scripts/eval_locally/eval_suite-template.sh`
"""

import json
from importlib import metadata
from os import error
from pathlib import Path
from types import SimpleNamespace

import click
import tqdm

from ..svg_glyph_gen_v2.filter_by_pangram_svg import blake2_hash
from ..svg_glyph_gen_v2.utils import (
    count_lines,
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
    "--input_infer_img_base64_dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    required=True,
)
@click.option(
    "--field",
    default="predict",
    required=True,
    help="JSON field containing SVG text.",
)
@click.option(
    "--output_dir",
    type=click.Path(),
    default=None,
)
@click.option("--overwrite", is_flag=True, default=False)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    eval_data_builder = EvalDataBuilder(args)
    eval_data_builder.run()


class EvalDataBuilder:
    SYSTEM_PROMPT = "You are an OCR engine. Read the image precisely."
    PROMPT = "Recognize the text in the image. Output exactly the recognized text once. Preserve case and accents as seen. Do not guess missing characters. If nothing readable is present, output exactly: No text recognized."

    def __init__(self, args):
        self.args = args

        input_infer_jsonl_dir = Path(args.input_infer_jsonl_dir)
        self.input_infer_jsonl_dir = input_infer_jsonl_dir
        self.input_infer_img_base64_dir = args.input_infer_img_base64_dir

        output_dir = args.output_dir
        if output_dir is None:
            output_dir = input_infer_jsonl_dir.parent / "ocr_eval_data"
        # prepare output dir and logger
        should_skip, logger = prepare_output_dir_and_logger(
            output_dir=output_dir,
            overwrite=args.overwrite,
        )
        if should_skip:
            exit()
        self.output_dir = Path(output_dir)
        self.logger = logger

    def run(self):
        input_infer_jsonl_dir = self.input_infer_jsonl_dir
        input_infer_img_base64_dir = self.input_infer_img_base64_dir

        num_samples_svg = count_lines(input_infer_jsonl_dir)
        num_samples_img_base64 = count_lines(input_infer_img_base64_dir)
        if num_samples_svg != num_samples_img_base64:
            error_msg = f"num_samples_svg {num_samples_svg} != num_samples_img_base64 {num_samples_img_base64}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # test single data
        infer_data = next(iter(load_jsonl_by_generator(input_infer_jsonl_dir)))
        img_base64_data = next(
            iter(load_jsonl_by_generator(input_infer_img_base64_dir))
        )
        alpaca_data = self.build_alpaca_data(infer_data, img_base64_data)

        output_jsonl_path = self.output_dir / "alpaca.jsonl"
        self.logger.info(f"Writing alpaca data to {output_jsonl_path}")

        result_list = []
        for infer_data, img_base64_data in tqdm.tqdm(
            zip(
                load_jsonl_by_generator(input_infer_jsonl_dir),
                load_jsonl_by_generator(input_infer_img_base64_dir),
            ),
            total=num_samples_svg,
        ):
            alpaca_data = self.build_alpaca_data(infer_data, img_base64_data)
            result_list.append(alpaca_data)

        write_jsonl(result_list, output_jsonl_path, logger=self.logger)

        # write dataset_info.json
        dataset_info = {
            "eval_data": {
                "file_name": output_jsonl_path.name,
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "response": "output",
                    "images": "images",
                },
            }
        }
        dataset_info_path = self.output_dir / "dataset_info.json"
        with open(dataset_info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)
        self.logger.info(f"Dataset info saved to {dataset_info_path}")

        #

    def build_alpaca_data(self, infer_data, img_base64_data):
        if self.args.field not in infer_data:
            error_msg = f"field '{self.args.field}' not found in infer_data"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        svg_text = infer_data.get(self.args.field)
        img_hash = blake2_hash(svg_text)
        img_hash_from_img_base64 = img_base64_data["img_hash"]

        if img_hash != img_hash_from_img_base64:
            error_msg = f"img_hash {img_hash} != img_hash_from_img_base64 {img_hash_from_img_base64}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        img_base64 = img_base64_data["img_base64"]

        metadata_str = infer_data["metadata"]
        metadata = json.loads(metadata_str)
        content_str = metadata["content_str"]

        alpaca_data = {
            "system": self.SYSTEM_PROMPT,
            "instruction": f"{self.PROMPT}",
            "output": content_str,
            "images": [img_base64],
            "metadata": metadata_str,
        }
        return alpaca_data


if __name__ == "__main__":
    main()
