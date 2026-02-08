"""
We render fonts with content "GgAa", and ue the LMM OCR results to filter out fonts that are not readable.

There are cases which are recognized as "Gg Aa", we choose to keep them.

We rename `google_font_metadata-filter_invalid-filter_by_pangram` to `.google_font_metadata-filter_invalid-filter_by_pangram`
to make sure we use the lmm ocr results for later processing.

python -m src.svg_glyph_gen_v2.filter_fonts_by_lmm_ocr \
    --input_gfont_metadata data/processed/.google_font_metadata-filter_invalid-filter_by_pangram \
    --input_lmm_ocr data/processed/filter_fonts_by_lmm_ocr/results_ocr_eval-Qwen2.5-VL-32B-Instruct-acc-use_case \
    --output_dir data/processed/google_font_metadata-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr \
"""

import json
from pathlib import Path
from typing import Union

import click
import tqdm

from .utils import load_jsonl, prepare_output_dir_and_logger, write_jsonl


def remove_whitespace_for_parsed_predict_and_regrade(line_dict):
    parsed_predict = line_dict["parsed_predict"]
    parsed_gt = line_dict["parsed_gt"]
    parsed_predict = "".join(parsed_predict.split())
    score = parsed_predict == parsed_gt

    line_dict["parsed_predict"] = parsed_predict
    line_dict["score"] = score


def compute_acc(line_dict_list, logger):
    num_correct = 0
    for line_dict in line_dict_list:
        num_correct += line_dict["score"]
    acc = num_correct / len(line_dict_list)
    return {
        "acc": acc,
        "num_correct": num_correct,
        "num_total": len(line_dict_list),
    }


@click.command()
@click.option("--input_gfont_metadata", type=click.Path(exists=True), required=True)
@click.option("--input_lmm_ocr", type=click.Path(exists=True), required=True)
@click.option("--output_dir", type=click.Path(), required=True)
@click.option("--overwrite", is_flag=True, default=False)
def main(
    input_gfont_metadata: str,
    input_lmm_ocr: str,
    output_dir: Union[str, Path],
    overwrite: bool,
):
    # prepare output dir and logger
    should_skip, logger = prepare_output_dir_and_logger(
        output_dir=output_dir,
        overwrite=overwrite,
    )
    if should_skip:
        exit()
    output_dir = Path(output_dir)

    metadata = load_jsonl(input_gfont_metadata, logger=logger)
    lmm_ocr_data = load_jsonl(input_lmm_ocr, logger=logger)

    logger.info(f"acc before remove whitespace:\t{compute_acc(lmm_ocr_data, logger)}")
    for line_dict in lmm_ocr_data:
        remove_whitespace_for_parsed_predict_and_regrade(line_dict)
    logger.info(f"acc after remove whitespace:\t{compute_acc(lmm_ocr_data, logger)}")

    identifier2metadata_idx = {}
    for idx, metadata_item in enumerate(metadata):
        identifier2metadata_idx[metadata_item["identifier"]] = idx
    ocr_identifier2lmm_ocr_idx = {}
    for idx, lmm_ocr_item in enumerate(lmm_ocr_data):
        lmm_ocr_metadata = json.loads(lmm_ocr_item["metadata"])
        ocr_identifier2lmm_ocr_idx[lmm_ocr_metadata["identifier"]] = idx

    len_metadata = len(identifier2metadata_idx.keys())
    len_ocr = len(ocr_identifier2lmm_ocr_idx.keys())
    if len_metadata > len_ocr:
        logger.warning(
            f"Number of identifiers in metadata is more than LMM OCR data: {len_metadata} vs {len_ocr}"
        )
    set_diff = set(ocr_identifier2lmm_ocr_idx.keys()) - set(
        identifier2metadata_idx.keys()
    )
    if len(set_diff) > 0:
        logger.error(
            f"Number of identifiers in metadata ({len(identifier2metadata_idx)}) does not match number of identifiers in LMM OCR data ({len(ocr_identifier2lmm_ocr_idx)})"
        )
        raise ValueError("Number of identifiers mismatch")

    filtered_metadata = []
    for identifier in tqdm.tqdm(ocr_identifier2lmm_ocr_idx, desc="Filtering metadata"):
        lmm_ocr_item = lmm_ocr_data[ocr_identifier2lmm_ocr_idx[identifier]]
        metadata_item = metadata[identifier2metadata_idx[identifier]]
        if lmm_ocr_item["score"] is True:
            filtered_metadata.append(metadata_item)
    len_before = len(metadata)
    len_after = len(filtered_metadata)
    logger.info(
        f"Filtered {len_before - len_after} out of {len_before} fonts -> {len_after}"
    )

    output_path = output_dir / "filtered_by_lmm_ocr_metadata.jsonl"
    write_jsonl(filtered_metadata, output_path, logger=logger)


if __name__ == "__main__":
    main()
