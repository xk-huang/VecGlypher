"""
# google fonts train vs google fonts test
python -m src.tools.check_data_contamination \
    -a data/processed_envato/filtered_sft/250903-alphanumeric/train_font_family \
    -b data/processed_envato/filtered_sft/250903-alphanumeric/ood_font_family

# google fonts train vs google fonts test
python -m src.tools.check_data_contamination \
    -a data/processed_envato/filtered_sft/250910-alphanumeric-abs_coord/train_font_family \
    -b data/processed_envato/filtered_sft/250910-alphanumeric-abs_coord/ood_font_family

# envato fonts train vs google fonts test
python -m src.tools.check_data_contamination \
    -a data/processed_envato/filtered_sft/250903-envato-alphanumeric/train_font_family \
    -b data/processed_envato/filtered_sft/250903-alphanumeric/ood_font_family

# envato fonts train vs google fonts test
python -m src.tools.check_data_contamination \
    -a data/processed_envato/filtered_sft/250903-envato-alphanumeric-abs_coord/train_font_family \
    -b data/processed_envato/filtered_sft/250910-alphanumeric-abs_coord/ood_font_family
"""

import json
import logging
import re

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
def main(input_jsonl_a, input_jsonl_b):
    logger.info(f"input_jsonl_a: {input_jsonl_a}")
    logger.info(f"input_jsonl_b: {input_jsonl_b}")
    data_a = load_jsonl(input_jsonl_a)
    data_b = load_jsonl(input_jsonl_b)

    filtered_data_a = filter_by_metadata_content_str(data_a)
    filtered_data_b = filter_by_metadata_content_str(data_b)

    svg_path_set_a = build_svg_hash_set(filtered_data_a)
    svg_path_set_b = build_svg_hash_set(filtered_data_b)

    overlap_svg_path_set = svg_path_set_a & svg_path_set_b
    len_overlap = len(overlap_svg_path_set)
    print(f"overlap_svg_path_set: {len_overlap}")

    len_data_a = len(svg_path_set_a)
    len_data_b = len(svg_path_set_b)
    overlap_over_a = len_overlap / len_data_a
    overlap_over_b = len_overlap / len_data_b
    print(f"overlap_ratio_a: {overlap_over_a:.6f} ({len_overlap} / {len_data_a})")
    print(f"overlap_ratio_b: {overlap_over_b:.6f} ({len_overlap} / {len_data_b})")


def build_svg_hash_set(data):
    svg_path_hash_set = set()
    num_duplicate_svg_hash = 0
    for d in tqdm.tqdm(data, desc="build svg path hash set"):
        svg = d["output"]
        paths = extract_svg_path_str(svg)
        paths_str = "\n".join(paths)
        svg_hash = blake2_hash(paths_str, digest_len=32)
        if svg_hash in svg_path_hash_set:
            num_duplicate_svg_hash += 1

        svg_path_hash_set.add(svg_hash)
    logger.info(f"num_duplicate_svg_hash: {num_duplicate_svg_hash}")
    logger.info(f"num of svg: {len(data)} -> {len(svg_path_hash_set)}")

    return svg_path_hash_set


def extract_svg_path_str(svg):
    # Regex pattern to extract path definitions
    pattern = r'<path[^>]*\sd="([^"]+)"'

    # Find all matches
    paths = re.findall(pattern, svg, flags=re.IGNORECASE | re.DOTALL)
    return paths


def filter_by_metadata_content_str(data, target_content_str=None):
    if target_content_str is None:
        logger.info("no filtering by content_str")
        return data

    logger.info(f"filtering by content_str: {target_content_str}")

    filtered_data = []
    for d in tqdm.tqdm(data, desc=f"filtering by content_str: {target_content_str}"):
        if "metadata" not in d:
            raise ValueError("metadata not found in data")
        metadata = d["metadata"]
        metadata = json.loads(metadata)
        content_str = metadata["content_str"]
        if content_str == target_content_str:
            filtered_data.append(d)
    logger.info(f"filtered: {len(data)} -> {len(filtered_data)}")

    return filtered_data


if __name__ == "__main__":
    main()
