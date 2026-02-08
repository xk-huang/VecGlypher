"""
python -m src.svg_glyph_gen_v2.validate_no_overlap_font_family \
-a data/processed/sft/250903-alphanumeric/train_font_family
-b data/processed/sft/250903-alphanumeric/ood_font_family
-o data/processed/sft/250903-alphanumeric/overlap_font_family
"""

import json
import shutil
from pathlib import Path

import click
import tqdm

from .utils import load_jsonl, prepare_output_dir_and_logger


@click.command()
@click.option(
    "--input_jsonl_1",
    "-a",
    type=click.Path(exists=True),
    # default="data/processed/sft/250903-alphanumeric/train_font_family",
    required=True,
)
@click.option(
    "--input_jsonl_2",
    "-b",
    type=click.Path(exists=True),
    # default="data/processed/sft/250903-alphanumeric/ood_font_family",
    required=True,
)
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(),
    # default="data/processed/sft/250903-alphanumeric/overlap_font_family",
    required=True,
)
@click.option("--overwrite", is_flag=True, default=False)
def main(
    input_jsonl_1: str,
    input_jsonl_2: str,
    output_dir: str,
    overwrite: bool,
):
    # prepare output dir and logger
    should_skip, logger = prepare_output_dir_and_logger(
        output_dir=output_dir,
        overwrite=overwrite,
    )
    output_dir = Path(output_dir)
    if should_skip:
        exit()
    font_family_set_1 = get_font_family_set(input_jsonl_1, logger)
    font_family_set_2 = get_font_family_set(input_jsonl_2, logger)

    len_font_family_set_1 = len(font_family_set_1)
    len_font_family_set_2 = len(font_family_set_2)
    logger.info(f"font family length {input_jsonl_1}: {len_font_family_set_1}")
    logger.info(f"font family length {input_jsonl_2}: {len_font_family_set_2}")

    overlap_font_family_set = font_family_set_1.intersection(font_family_set_2)
    len_font_family_set = len(overlap_font_family_set)
    logger.info(f"font family overlap length: {len_font_family_set}")

    output_overlap_font_family_path = output_dir / "overlap_font_family.txt"
    with open(output_overlap_font_family_path, "w") as f:
        for font_family in overlap_font_family_set:
            f.write(f"{font_family}\n")
    logger.info(f"overlap font family saved to {output_overlap_font_family_path}")


def get_font_family_set(input_jsonl, logger):
    font_family_set = set()
    for line in tqdm.tqdm(load_jsonl(input_jsonl, logger)):
        metadata = line["metadata"]
        metadata = json.loads(metadata)

        font_family_dir_name = metadata["font_family_dir_name"]
        font_family_set.add(font_family_dir_name)

    return font_family_set


if __name__ == "__main__":
    main()
