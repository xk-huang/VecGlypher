"""
# google fonts train vs google fonts test
python -m src.tools.decontaminate_data \
    -a data/processed_envato/filtered_sft/250903-alphanumeric/train_font_family \
    -b data/processed_envato/filtered_sft/250903-alphanumeric/ood_font_family
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import click
import tqdm

from ..svg_glyph_gen_v2.filter_by_pangram_svg import blake2_hash
from ..svg_glyph_gen_v2.utils import load_jsonl, write_jsonl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command(
    help="Compare two jsonl files. Two modes: hash, lf_sft, by default lf_sft"
)
@click.option(
    "--input_reference_jsonl", "-a", type=click.Path(exists=True), required=True
)
@click.option(
    "--input_candidate_jsonl", "-b", type=click.Path(exists=True), required=True
)
def main(input_reference_jsonl, input_candidate_jsonl):
    logger.info(f"input_reference_jsonl: {input_reference_jsonl}")
    logger.info(f"input_candidate_jsonl: {input_candidate_jsonl}")
    ref_data = load_jsonl(input_reference_jsonl)
    cand_data = load_jsonl(input_candidate_jsonl)

    # build hash set
    ref_hash_set, _ = build_svg_hash_set_and_idx_map(ref_data)
    cand_hash_set, svg_path_hash2idx = build_svg_hash_set_and_idx_map(cand_data)

    overlap_svg_path_set = ref_hash_set & cand_hash_set
    len_overlap = len(overlap_svg_path_set)
    logger.info(f"overlap_svg_path_set: {len_overlap}")

    len_ref_data = len(ref_hash_set)
    len_cand_data = len(cand_hash_set)
    overlap_over_ref = len_overlap / len_ref_data
    overlap_over_cand = len_overlap / len_cand_data
    logger.info(
        f"overlap_ratio_ref: {overlap_over_ref:.6f} ({len_overlap} / {len_ref_data})"
    )
    logger.info(
        f"overlap_ratio_cand: {overlap_over_cand:.6f} ({len_overlap} / {len_cand_data})"
    )

    if len_overlap == 0:
        logger.info("no overlap, return")
        return

    # decontaminate candidate data
    decon_cand_hash_set = cand_hash_set - overlap_svg_path_set
    len_cand = len(cand_data)
    len_decon_cand = len(decon_cand_hash_set)
    logger.info(
        f"candidate hash len: {len_cand} -> {len_decon_cand} (remove {len_cand - len_decon_cand} repeat samples)"
    )

    decon_cand_idx = []
    for svg_hash in decon_cand_hash_set:
        decon_cand_idx.extend(svg_path_hash2idx[svg_hash])
    len_cand = len(cand_data)
    len_decon_cand = len(decon_cand_idx)
    logger.info(
        f"decontaminate candidate samples: {len_cand} -> {len_decon_cand} (remove {len_cand - len_decon_cand} samples)"
    )

    # save decontaminate candidate data
    input_candidate_jsonl = Path(input_candidate_jsonl)
    output_candidate_jsonl = (
        input_candidate_jsonl.parent / f"{input_candidate_jsonl.name}_decon"
    )
    if output_candidate_jsonl.exists():
        err_msg = f"output_candidate_jsonl already exists: {output_candidate_jsonl}, remove manually."
        logger.error(err_msg)
        raise ValueError(err_msg)

    decon_cand_data = [cand_data[i] for i in decon_cand_idx]
    logger.info(f"decontaminate candidate data: {len(decon_cand_data)}")
    logger.info(f"output_candidate_jsonl: {output_candidate_jsonl}")
    write_jsonl(decon_cand_data, output_candidate_jsonl / "data.jsonl", logger=logger)


def build_svg_hash_set_and_idx_map(data):
    svg_path_hash_set = set()
    hash2idx = defaultdict(list)

    num_duplicate_svg_hash = 0
    for idx, d in enumerate(tqdm.tqdm(data, desc="build svg path hash set")):
        svg = d["output"]
        paths = extract_svg_path_str(svg)
        paths_str = "\n".join(paths)
        svg_hash = blake2_hash(paths_str, digest_len=32)

        if svg_hash in svg_path_hash_set:
            num_duplicate_svg_hash += 1

        svg_path_hash_set.add(svg_hash)
        hash2idx[svg_hash].append(idx)

    logger.info(f"num_duplicate_svg_hash: {num_duplicate_svg_hash}")
    len_data = len(data)
    len_uniq = len(svg_path_hash_set)
    logger.info(f"num of svg: {len_data} -> {len_uniq} (remove {len_data - len_uniq})")
    return svg_path_hash_set, hash2idx


def extract_svg_path_str(svg):
    # Regex pattern to extract path definitions
    pattern = r'<path[^>]*\sd="([^"]+)"'

    # Find all matches
    paths = re.findall(pattern, svg, flags=re.IGNORECASE | re.DOTALL)
    return paths


if __name__ == "__main__":
    main()
