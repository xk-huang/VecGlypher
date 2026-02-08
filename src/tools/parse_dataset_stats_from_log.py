"""
# Metadata

extract_metadata_tags:
- data/processed/google_font_metadata
- (convert_to_gfont_format) data/processed_envato/metadata_in_gfont

filter_invalid_fonts
- data/processed/google_font_metadata-filter_invalid
- data/processed_envato/metadata_in_gfont-filter_invalid

filter_by_pangram_svg
- data/processed/.google_font_metadata-filter_invalid-filter_by_pangram
- data/processed_envato/.metadata_in_gfont-filter_invalid-filter_by_pangram



Use "GgAa" to filter unrecognized or unicase fonts
filter_fonts_by_lmm_ocr
- data/processed/google_font_metadata-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr
- data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr



# Metadata stats:
(unused) render_pangram_for_fonts
- data/processed/google_font_metadata-filter_invalid-pangram/hist
- data/processed_envato/metadata_in_gfont-filter_invalid-pangram/hist

(unused) stat_field_values
- data/processed/google_font_metadata-stat_field_values
- data/processed_envato/metadata_in_gfont-stat_field_values

(unused) stat_font_vertical
- data/processed/google_font_metadata-stat_font_vertical
- data/processed_envato/metadata_in_gfont-stat_font_vertical



# Content
(unused) src.svg_glyph_gen_v2.gather_content
- data/processed/content
- data/processed_envato/content



# Dataset split
split_train_test_index_v2
- data/processed/split_train_test_index/alphanumeric
    - data/processed/split_train_test_index/alphanumeric/content/logs/split_train_test_index_v2.log
    - data/processed/split_train_test_index/alphanumeric/font_family/logs/split_train_test_index_v2.log
- data/processed_envato/split_train_test_index/alphanumeric
    - data/processed_envato/split_train_test_index/alphanumeric/content/logs/split_train_test_index_v2.log
    - data/processed_envato/split_train_test_index/alphanumeric/font_family/logs/split_train_test_index_v2.log



# Dataset stats
- (unused) data/processed/sft/250903-alphanumeric/stat_token_len/train_font_family/250903-alphanumeric-train_font_family.pdf
- (unused) data/processed/sft/250903-alphanumeric/stat_token_len/ood_font_family/250903-alphanumeric-ood_font_family.pdf
- data/processed/sft/250903-alphanumeric/dataset_stat.json
- data/processed/filtered_sft/250903-alphanumeric/dataset_stat.json

[NOTE] The date of google fonts alphanumeric-abs_coord is different (the only one): 250910 instead of 250903
- (unused) data/processed/sft/250910-alphanumeric-abs_coord/stat_token_len/train_font_family/250910-alphanumeric-abs_coord-train_font_family.pdf
- (unused) data/processed/sft/250910-alphanumeric-abs_coord/stat_token_len/ood_font_family/250910-alphanumeric-abs_coord-ood_font_family.pdf
- data/processed/sft/250910-alphanumeric-abs_coord/dataset_stat.json
- data/processed/filtered_sft/250910-alphanumeric-abs_coord/dataset_stat.json

- (unused) data/processed_envato/sft/250903-envato-alphanumeric/stat_token_len/train_font_family/250903-envato-alphanumeric-train_font_family.pdf
- (unused) data/processed_envato/sft/250903-envato-alphanumeric/stat_token_len/ood_font_family/250903-envato-alphanumeric-ood_font_family.pdf
- data/processed_envato/sft/250903-envato-alphanumeric/dataset_stat.json
- data/processed_envato/filtered_sft/250903-envato-alphanumeric/dataset_stat.json

- (unused) data/processed_envato/sft/250903-envato-alphanumeric-abs_coord/stat_token_len/train_font_family/250903-envato-alphanumeric-abs_coord-train_font_family.pdf
- (unused) data/processed_envato/sft/250903-envato-alphanumeric-abs_coord/stat_token_len/ood_font_family/250903-envato-alphanumeric-abs_coord-ood_font_family.pdf
- data/processed_envato/sft/250903-envato-alphanumeric-abs_coord/dataset_stat.json
- data/processed_envato/filtered_sft/250903-envato-alphanumeric-abs_coord/dataset_stat.json
"""

import ast
import json
import logging
import pprint
import re
from functools import partial
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GOOGLE_FONTS_LOG_PATHS = {
    # metadata
    "extract_metadata_tags": "data/processed/google_font_metadata/logs/extract_metadata_tags.log",
    "filter_invalid_fonts": "data/processed/google_font_metadata-filter_invalid/logs/filter_invalid_fonts.log",
    "filter_by_pangram_svg": "data/processed/.google_font_metadata-filter_invalid-filter_by_pangram/logs/filter_by_pangram_svg.log",
    "filter_fonts_by_lmm_ocr": "data/processed/google_font_metadata-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr/logs/filter_fonts_by_lmm_ocr.log",
    # dataset split
    "split_train_test_index_v2_content": "data/processed/split_train_test_index/alphanumeric/content/logs/split_train_test_index_v2.log",
    "split_train_test_index_v2_font": "data/processed/split_train_test_index/alphanumeric/font_family/logs/split_train_test_index_v2.log",
    # dataset stats
    "sft_data_stats": "data/processed/sft/250903-alphanumeric/dataset_stat.json",
    "filterd_sft_data_stats": "data/processed/filtered_sft/250903-alphanumeric/dataset_stat.json",
    "sft_data_stats_abs_coord": "data/processed/sft/250910-alphanumeric-abs_coord/dataset_stat.json",
    "filterd_sft_data_stats_abs_coor": "data/processed/filtered_sft/250910-alphanumeric-abs_coord/dataset_stat.json",
}

ENVATO_FONTS_LOG_PATHS = {
    # metadata
    "extract_metadata_tags": "data/processed_envato/metadata_in_gfont/logs/extract_metadata_tags.log",
    "filter_invalid_fonts": "data/processed_envato/metadata_in_gfont-filter_invalid/logs/filter_invalid_fonts.log",
    "filter_by_pangram_svg": "data/processed_envato/.metadata_in_gfont-filter_invalid-filter_by_pangram/logs/filter_by_pangram_svg.log",
    "filter_fonts_by_lmm_ocr": "data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr/logs/filter_fonts_by_lmm_ocr.log",
    # dataset split
    "split_train_test_index_v2_content": "data/processed_envato/split_train_test_index/alphanumeric/content/logs/split_train_test_index_v2.log",
    "split_train_test_index_v2_font": "data/processed_envato/split_train_test_index/alphanumeric/font_family/logs/split_train_test_index_v2.log",
    # dataset stats
    "sft_data_stats": "data/processed_envato/sft/250903-envato-alphanumeric/dataset_stat.json",
    "filterd_sft_data_stats": "data/processed_envato/filtered_sft/250903-envato-alphanumeric/dataset_stat.json",
    "sft_data_stats_abs_coord": "data/processed_envato/sft/250903-envato-alphanumeric-abs_coord/dataset_stat.json",
    "filterd_sft_data_stats_abs_coor": "data/processed_envato/filtered_sft/250903-envato-alphanumeric-abs_coord/dataset_stat.json",
}


def extract_metadata_tags(log_path):
    """
    num_success_fonts: 3403" and "num_failed_fonts: 109
    """
    with open(log_path, "r") as f:
        log = f.read()

    def _search_and_return_number(pattern, log, log_path):
        # NOTE: only return the first match
        re_num = re.search(pattern, log)
        if re_num is None:
            err_msg = f"Failed to extract num_success_fonts from {log_path}."
            logger.error(err_msg)
            raise ValueError(err_msg)
        try:
            num = int(re_num.group(1))
        except ValueError:
            err_msg = f"Failed to convert num_success_fonts to int: {re_num.group(1)}."
            logger.error(err_msg)
            raise ValueError(err_msg)
        return num

    pattern = r"num_success_fonts: (\d+)"
    num_success_fonts = _search_and_return_number(pattern, log, log_path)
    pattern = r"num_failed_fonts: (\d+)"
    num_failed_fonts = _search_and_return_number(pattern, log, log_path)

    return {
        "extract_metadata_tags": {
            "num_success_fonts": num_success_fonts,
            "num_failed_fonts": num_failed_fonts,
        }
    }


def filter_invalid_fonts(log_path):
    """
    Number of fonts: 3403 -> 3322
    """
    with open(log_path, "r") as f:
        log = f.read()

    pattern = r"Number of fonts: (\d+) -> (\d+)"
    re_num = re.search(pattern, log)
    if re_num is None:
        err_msg = f"Failed to extract num_success_fonts from {log_path}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    try:
        num_before = int(re_num.group(1))
    except ValueError:
        err_msg = f"Failed to convert num_success_fonts to int: {re_num.group(1)}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    try:
        num_after = int(re_num.group(2))
    except ValueError:
        err_msg = f"Failed to convert num_success_fonts to int: {re_num.group(2)}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    return {
        "filter_invalid_fonts": {
            "num_fonts_before_invalid_filter": num_before,
            "num_fonts_after_invalid_filter": num_after,
        }
    }


def filter_by_pangram_svg(log_path):
    """
    [Deduplication] Number of fonts: 2989 -> 2645
    """
    with open(log_path, "r") as f:
        log = f.read()

    pattern = r"\[Deduplication\] Number of fonts: (\d+) -> (\d+)"
    re_num = re.search(pattern, log)
    if re_num is None:
        err_msg = f"Failed to extract num_success_fonts from {log_path}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    try:
        num_before = int(re_num.group(1))
    except ValueError:
        err_msg = f"Failed to convert num_success_fonts to int: {re_num.group(1)}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    try:
        num_after = int(re_num.group(2))
    except ValueError:
        err_msg = f"Failed to convert num_success_fonts to int: {re_num.group(2)}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    return {
        "filter_by_pangram_svg": {
            "num_fonts_before_pangram_filter": num_before,
            "num_fonts_after_pangram_filter": num_after,
        }
    }


def filter_fonts_by_lmm_ocr(log_path):
    """
    Filtered 148 out of 2645 fonts -> 2497
    """
    with open(log_path, "r") as f:
        log = f.read()

    pattern = r"Filtered (\d+) out of (\d+) fonts -> (\d+)"
    re_num = re.search(pattern, log)
    if re_num is None:
        err_msg = f"Failed to extract num_success_fonts from {log_path}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    try:
        num_before = int(re_num.group(2))
    except ValueError:
        err_msg = f"Failed to convert num_success_fonts to int: {re_num.group(2)}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    try:
        num_after = int(re_num.group(3))
    except ValueError:
        err_msg = f"Failed to convert num_success_fonts to int: {re_num.group(3)}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    return {
        "filter_fonts_by_lmm_ocr": {
            "num_fonts_before_lmm_ocr_filter": num_before,
            "num_fonts_after_lmm_ocr_filter": num_after,
        }
    }


def split_train_test_index_v2_content(log_path):
    """
    Loaded 64 content
    split_arg: {'total': 64, 'train': 64, 'ind_test': 10, 'ood_test': 0, 'dev': 10}
    """
    with open(log_path, "r") as f:
        log = f.read()

    pattern = r"Loaded (\d+) content"
    re_num = re.search(pattern, log)
    if re_num is None:
        err_msg = f"Failed to extract num_success_fonts from {log_path}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    try:
        num = int(re_num.group(1))
    except ValueError:
        err_msg = f"Failed to convert num_success_fonts to int: {re_num.group(1)}."
        logger.error(err_msg)
        raise ValueError(err_msg)

    pattern = r"split_arg: (\{.*\})"
    re_dict = re.search(pattern, log)
    if re_dict is None:
        err_msg = f"Failed to extract split_arg from {log_path}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    try:
        result_dict = re_dict.group(1)
        result_dict = ast.literal_eval(result_dict)
    except ValueError:
        err_msg = f"Failed to convert num_success_fonts to int: {re_dict.group(1)}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    return {
        "split_train_test_index_v2_content": {
            "num_content": num,
            "content_split": result_dict,
        }
    }


def split_train_test_index_v2_font(log_path):
    """
    Loaded 2497 records
    split_arg for font family: {'total': 1117, 'train': 997, 'ind_test': 120, 'ood_test': 120, 'dev': 10}
    stats_dict: {'train': 2243, 'ind_test': 234, 'ood_test': 254, 'dev': 41}
    """
    with open(log_path, "r") as f:
        log = f.read()

    pattern = r"Loaded (\d+) records"
    re_num = re.search(pattern, log)
    if re_num is None:
        err_msg = f"Failed to extract num_success_fonts from {log_path}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    try:
        num = int(re_num.group(1))
    except ValueError:
        err_msg = f"Failed to convert num_success_fonts to int: {re_num.group(1)}."
        logger.error(err_msg)
        raise ValueError(err_msg)

    pattern = r"split_arg for font family: (\{.*\})"
    re_dict = re.search(pattern, log)
    if re_dict is None:
        err_msg = f"Failed to extract split_arg from {log_path}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    try:
        input_font_family_split_dict = re_dict.group(1)
        input_font_family_split_dict = ast.literal_eval(input_font_family_split_dict)
    except ValueError:
        err_msg = f"Failed to convert num_success_fonts to int: {re_dict.group(1)}."
        logger.error(err_msg)
        raise ValueError(err_msg)

    pattern = r"stats_dict: (\{.*\})"
    re_dict = re.search(pattern, log)
    if re_dict is None:
        err_msg = f"Failed to extract split_arg from {log_path}."
        logger.error(err_msg)
        raise ValueError(err_msg)
    try:
        font_split_dict = re_dict.group(1)
        font_split_dict = ast.literal_eval(font_split_dict)
    except ValueError:
        err_msg = f"Failed to convert num_success_fonts to int: {re_dict.group(1)}."
        logger.error(err_msg)
        raise ValueError(err_msg)

    return {
        "split_train_test_index_v2_font": {
            "num_fonts": num,
            "input_font_family_split": input_font_family_split_dict,
            "font_split": font_split_dict,
        }
    }


def load_dataset_stats_json(log_path, key):
    with open(log_path, "r") as f:
        data = json.load(f)
    new_data = {}
    for k, v in data.items():
        v = v["num_samples"]
        new_data[k] = v
    return {key: new_data}


FUNCTIONS = {
    "extract_metadata_tags": extract_metadata_tags,
    "filter_invalid_fonts": filter_invalid_fonts,
    "filter_by_pangram_svg": filter_by_pangram_svg,
    "filter_fonts_by_lmm_ocr": filter_fonts_by_lmm_ocr,
    "split_train_test_index_v2_content": split_train_test_index_v2_content,
    "split_train_test_index_v2_font": split_train_test_index_v2_font,
    "sft_data_stats": partial(load_dataset_stats_json, key="sft_data_stats"),
    "filterd_sft_data_stats": partial(
        load_dataset_stats_json, key="filterd_sft_data_stats"
    ),
    "sft_data_stats_abs_coord": partial(
        load_dataset_stats_json, key="sft_data_stats_abs_coord"
    ),
    "filterd_sft_data_stats_abs_coor": partial(
        load_dataset_stats_json, key="filterd_sft_data_stats_abs_coor"
    ),
}


def parse_dataset_log_from_files(log_path_dict):
    results = {}
    for log_name, log_path in log_path_dict.items():
        logger.info(f"Processing {log_name}: {log_path}...")
        log_path = Path(log_path)
        if not log_path.exists():
            logger.warning(f"{log_path} does not exist.")
            continue

        function = FUNCTIONS.get(log_name, None)
        if function is None:
            err_msg = f"Function is not defined\n{log_name}: {log_path} "
            logger.error(err_msg)
            raise ValueError(err_msg)

        result = function(log_path)
        results.update(result)
    logger.info(f"Finished processing {len(results)} logs.")
    logger.info(f"Results:\n{pprint.pformat(results)}")
    return results


@click.command()
@click.argument("output_dir", type=str, default="data/dataset_stats_from_log")
def main(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    output_path = output_dir / "dataset_stats_from_log-google_fonts.json"
    result = parse_dataset_log_from_files(GOOGLE_FONTS_LOG_PATHS)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved to {output_path}")

    output_path = output_dir / "dataset_stats_from_log-envato_fonts.json"
    result = parse_dataset_log_from_files(ENVATO_FONTS_LOG_PATHS)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
