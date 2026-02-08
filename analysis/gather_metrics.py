#!/usr/bin/env python3
"""
Script to gather metrics from JSON files in the outputs directory and save to TSV format.

python analysis/gather_metrics.py [INPUT_DIR] [OUTPUT_DIR] [-p PARTS_SEP_BY_COMMA]
"""

import csv
import glob
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import click
import numpy as np


def find_metric_files(input_dir):
    """
    Find all metric files matching the specified patterns.

    """
    patterns = ["**/results*/*.json"]

    found_files = {}

    # XXX: entangled with the function flow in `gather_metrics`.
    METRIC_DIR_NAME2METRIC_NAME = {
        "results_img_eval/": "img_eval",
        "results_ocr_eval-use_case/": "ocr_w_case",
        "results_ocr_eval-no_use_case/": "ocr_wo_case",
        "results_point_cloud_eval/": "chamfer",
        "results_point_cloud_eval-align_pcd/": "chamfer_aligned_wo_scale",
        "results_point_cloud_eval-align_pcd-estimate_scale/": "chamfer_aligned_w_scale",
    }

    for pattern in patterns:
        full_pattern = os.path.join(input_dir, pattern)
        print(f"Searching for files matching pattern: {full_pattern}")
        files = glob.iglob(full_pattern, recursive=True)

        for file_path in files:
            # Extract the base directory path (the ** part)
            base_dir = str(get_rel_exp_job_dir(file_path, input_dir))

            if base_dir not in found_files:
                found_files[base_dir] = {}

            # NOTE: organize those metric files to dict, then load them and add to df.
            found = False
            for metric_dir_name in METRIC_DIR_NAME2METRIC_NAME:
                if metric_dir_name in file_path:
                    metric_name = METRIC_DIR_NAME2METRIC_NAME[metric_dir_name]
                    print(file_path, "->", metric_name)
                    found_files[base_dir][metric_name] = file_path
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Unknown file: {file_path}, need to update the script."
                )
    print(f"Found {len(found_files)} directories with metric files")
    return found_files


def get_rel_exp_job_dir(file_path, input_dir):
    """
    The data structure must be as follows
    `eval_dir/result_file.json`
    """
    exp_job_dir = Path(file_path).parent.parent
    rel_exp_job_dir = exp_job_dir.relative_to(input_dir)
    return rel_exp_job_dir


def load_json_file(file_path):
    """Load JSON file and return data."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def gather_metrics(input_dir, output_dir, split_path_part=None):
    """Gather metrics from all found files and save to TSV."""
    found_files = find_metric_files(input_dir)

    if not found_files:
        print("No metric files found!")
        return

    # NOTE: each exp job is a dict in `all_data`
    # we gathe columns from all dicts. For fear of missing some columns, we add all columns to a list.
    all_data = []
    all_columns = dict()

    if split_path_part is not None:
        split_path_part = [int(x) for x in split_path_part.split(",")]

    for base_dir, files in found_files.items():
        row_data = {"path": base_dir}
        all_columns["path"] = None

        if split_path_part is not None:
            for split_path_level_idx in split_path_part:
                column_name = f"path_part_{split_path_level_idx}"
                row_data[column_name] = Path(base_dir).parts[split_path_level_idx]
                all_columns[column_name] = None

        # Load OCR evaluation metrics
        for column_name in ["ocr_w_case", "ocr_wo_case"]:
            all_columns[column_name] = None
            if column_name not in files:
                row_data[column_name] = np.nan
                continue

            ocr_data = load_json_file(files[column_name])
            if ocr_data and "accuracy" in ocr_data:
                row_data[column_name] = ocr_data["accuracy"]

        # Load Chamfer evaluation metrics
        # XXX: entangled with METRIC_DIR_NAME2METRIC_NAME.
        for column_name in [
            "chamfer",
            "chamfer_aligned_wo_scale",
            "chamfer_aligned_w_scale",
        ]:
            all_columns[column_name] = None
            if column_name not in files:
                row_data[column_name] = np.nan
                continue

            chamfer_data = load_json_file(files[column_name])
            if chamfer_data and "chamfer_distance" in chamfer_data:
                row_data[column_name] = chamfer_data["chamfer_distance"]

        # Load image evaluation metrics
        if "img_eval" in files:
            img_data = load_json_file(files["img_eval"])
            if img_data:
                # Add all keys from results_avg.json
                for column_name, value in img_data.items():
                    # Handle infinity values
                    if value == float("inf"):
                        value = "Infinity"
                    row_data[column_name] = value
                    all_columns[column_name] = None
        all_data.append(row_data)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "metrics.tsv")

    all_columns = list(all_columns.keys())

    # Write TSV file
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, delimiter="\t")
        writer.writeheader()

        for row in all_data:
            # Fill missing values with empty string
            complete_row = {col: row.get(col, np.nan) for col in all_columns}
            writer.writerow(complete_row)

    print(f"Metrics gathered successfully!")
    print(f"Output file: {output_file}")
    print(f"Found {len(all_data)} directories with metrics")
    print(f"Columns: {', '.join(all_columns)}")

    # Copy the output file to the input directory
    output_file_for_input_dir = os.path.join(input_dir, "metrics.tsv")
    shutil.copyfile(output_file, output_file_for_input_dir)
    print(f"Output file copied to input directory: {output_file_for_input_dir}")


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False), default="outputs/"
)
@click.argument("output_dir", type=str, default="outputs/gathered")
@click.option(
    "--split_path_part", "-p", type=str, default=None, help="split path parts, `-2,-1`"
)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    if not os.path.exists(args.input_dir):
        print(f"Input directory {args.input_dir} does not exist!")
        return

    gather_metrics(args.input_dir, args.output_dir, args.split_path_part)


if __name__ == "__main__":
    main()
