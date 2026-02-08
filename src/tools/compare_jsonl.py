"""
python -m src.tools.compare_jsonl \
    -a data/processed/filtered_sft/250903-alphanumeric/ood_font_family \
    -b /home/vecglypher/mnt/workspace/svg_glyph_llm/data/processed/filtered_sft/250903-alphanumeric/ood_font_family
"""

import json
import logging
from random import choice

import click
import tqdm

from ..svg_glyph_gen_v2.filter_by_pangram_svg import blake2_hash
from ..svg_glyph_gen_v2.utils import load_jsonl


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command(
    help="Compare two jsonl files. Two modes: hash, lf_sft, by default lf_sft"
)
@click.option("--input_jsonl_a", "-a", type=click.Path(exists=True), required=True)
@click.option("--input_jsonl_b", "-b", type=click.Path(exists=True), required=True)
@click.option(
    "--compare_type",
    "-t",
    type=click.Choice(["hash", "lf_sft"]),
    default="lf_sft",
)
def main(input_jsonl_a, input_jsonl_b, compare_type):
    logger.info(f"compare_type: {compare_type}")
    logger.info(f"input_jsonl_a: {input_jsonl_a}")
    logger.info(f"input_jsonl_b: {input_jsonl_b}")

    if compare_type == "hash":
        decode_jsonl = False
        data_a = load_jsonl(input_jsonl_a, decode_jsonl=decode_jsonl, logger=logger)
        data_b = load_jsonl(input_jsonl_b, decode_jsonl=decode_jsonl, logger=logger)
        if len(data_a) != len(data_b):
            raise ValueError(
                f"len(data_a) != len(data_b): {len(data_a)} != {len(data_b)}"
            )

        hash_a = {blake2_hash(d) for d in data_a}
        hash_b = {blake2_hash(d) for d in data_b}

        if hash_a == hash_b:
            logger.info("Congrats! Validation passed with hash")
        else:
            logger.error("Different")
            logger.error(f"hash_a: {len(hash_a)}")
            logger.error(f"hash_b: {len(hash_b)}")
            logger.error(f"hash_a - hash_b: {len(hash_a - hash_b)}")
            logger.error(f"hash_b - hash_a: {len(hash_b - hash_a)}")
            raise ValueError("Validation failed with hash")

    elif compare_type == "lf_sft":
        # NOTE: specialized comparison for llama-factory sft data
        decode_jsonl = True
        data_a = load_jsonl(input_jsonl_a, decode_jsonl=decode_jsonl, logger=logger)
        data_b = load_jsonl(input_jsonl_b, decode_jsonl=decode_jsonl, logger=logger)
        if len(data_a) != len(data_b):
            raise ValueError(
                f"len(data_a) != len(data_b): {len(data_a)} != {len(data_b)}"
            )

        # sort the data by identifier
        logger.info(f"Sorting data_a...")
        sorted_data_a = sort_sft_data(data_a)
        logger.info(f"Sorting data_b...")
        sorted_data_b = sort_sft_data(data_b)

        logger.info(f"Validating data_a & data_b...")
        VALIDATION_KEYS = ["output"]
        METADATA_VALIDATION_KEYS = ["identifier", "content_str"]
        total_len = len(sorted_data_a)
        for sorted_data_a_item, sorted_data_b_item in tqdm.tqdm(
            zip(sorted_data_a, sorted_data_b), total=total_len
        ):
            for validation_key in VALIDATION_KEYS:
                value_a = sorted_data_a_item[validation_key]
                value_b = sorted_data_b_item[validation_key]
                if value_a != value_b:
                    raise ValueError(
                        f"key `{validation_key}` mismatch:\n\ta: {value_a}\n\tb: {value_b}"
                    )

            metadata_a = json.loads(sorted_data_a_item["metadata"])
            metadata_b = json.loads(sorted_data_b_item["metadata"])
            for metadata_validation_key in METADATA_VALIDATION_KEYS:
                value_a = metadata_a[metadata_validation_key]
                value_b = metadata_b[metadata_validation_key]
                if value_a != value_b:
                    raise ValueError(
                        f"key `{metadata_validation_key}` mismatch:\n\ta: {value_a}\n\tb: {value_b}"
                    )

        logger.info(
            f"Congrats! Validation passed for keys: {VALIDATION_KEYS} & {METADATA_VALIDATION_KEYS}"
        )


def _sort_fn(x):
    metadata = json.loads(x["metadata"])
    identifier = metadata["identifier"]
    content_str = metadata["content_str"]
    return identifier, content_str


def sort_sft_data(data):
    return sorted(data, key=_sort_fn)


if __name__ == "__main__":
    main()
