#!/usr/bin/env python3
"""
Script to extract metadata of font files from Envato datasets.

This script:
1. Reads file-level and zip-level metadata CSV files
2. Filters for TTF files only
3. Merges data based on ITEM_ID
4. Validates matching keys have same values
5. Outputs to JSONL format


python -m src.svg_glyph_gen_v2_envato.extract_metadata
"""

import json
import logging
import os
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from ..svg_glyph_gen_v2.utils import prepare_output_dir_and_logger

# Set up logging
logger = logging.getLogger(__name__)


def load_metadata_files(
    file_metadata_path: str, zip_metadata_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the metadata CSV files."""
    logger.info(f"Loading file metadata from: {file_metadata_path}")
    file_metadata = pd.read_csv(file_metadata_path)
    logger.info(f"Loaded {len(file_metadata)} rows from file metadata")

    logger.info(f"Loading zip metadata from: {zip_metadata_path}")
    zip_metadata = pd.read_csv(zip_metadata_path)
    logger.info(f"Loaded {len(zip_metadata)} rows from zip metadata")

    return file_metadata, zip_metadata


def filter_ttf_files(file_metadata: pd.DataFrame) -> pd.DataFrame:
    """Filter file metadata to keep only TTF files."""
    logger.info("Filtering for TTF files only")
    ttf_files = file_metadata[file_metadata["FILE_TYPE"] == "ttf"].copy()
    logger.info(
        f"Found {len(ttf_files)} TTF files out of {len(file_metadata)} total files"
    )

    # remove font file without MD5
    ttf_files = ttf_files[~ttf_files["MD5"].isna()]
    logger.info(
        f"Found {len(ttf_files)} TTF files with MD5 out of {len(file_metadata)} total files"
    )
    return ttf_files


def validate_matching_keys(file_row: pd.Series, zip_row: pd.Series) -> bool:
    """Validate that matching keys between file and zip metadata have the same values."""
    file_keys = set(file_row.index)
    zip_keys = set(zip_row.index)
    common_keys = file_keys.intersection(zip_keys)

    mismatches = []
    for key in common_keys:
        file_val = file_row[key]
        zip_val = zip_row[key]

        # Handle NaN values - consider them equal if both are NaN
        if pd.isna(file_val) and pd.isna(zip_val):
            continue
        elif file_val != zip_val:
            mismatches.append(f"{key}: file='{file_val}' vs zip='{zip_val}'")

    if mismatches:
        logger.warning(
            f"Validation mismatches for ITEM_ID {file_row.get('ITEM_ID', 'unknown')}: {'; '.join(mismatches)}"
        )
        return False

    return True


def merge_metadata(ttf_files: pd.DataFrame, zip_metadata: pd.DataFrame) -> list[dict]:
    """Merge file and zip metadata based on ITEM_ID."""
    logger.info("Merging file and zip metadata based on ITEM_ID")

    # Create a dictionary for faster zip metadata lookup
    zip_dict = zip_metadata.set_index("ITEM_ID").to_dict("index")

    merged_data = []
    validation_failures = 0
    missing_zip_data = 0

    for _, file_row in tqdm(ttf_files.iterrows(), total=len(ttf_files)):
        item_id = file_row["ITEM_ID"]

        if item_id not in zip_dict:
            logger.warning(f"No zip metadata found for ITEM_ID: {item_id}")
            missing_zip_data += 1
            # Still include the file data even if zip data is missing
            merged_record = file_row.to_dict()
        else:
            zip_row = pd.Series(zip_dict[item_id])

            # Validate matching keys
            if not validate_matching_keys(file_row, zip_row):
                validation_failures += 1

            # Merge the data - file data takes precedence for overlapping keys
            merged_record = zip_row.to_dict()
            merged_record.update(file_row.to_dict())

        # Convert any NaN values to None for JSON serialization
        merged_record = {
            k: (None if pd.isna(v) else v) for k, v in merged_record.items()
        }
        merged_data.append(merged_record)

    logger.info(f"Merged {len(merged_data)} records")
    logger.info(f"Validation failures: {validation_failures}")
    logger.info(f"Missing zip data: {missing_zip_data}")

    return merged_data


def save_to_jsonl(data: list[dict], output_dir: str, chunk_size: int = 5000) -> None:
    """Save the merged data to JSONL format in chunks."""
    output_path = Path(output_dir) / "metadata.jsonl"
    logger.info(
        f"Saving {len(data)} records to: {output_path} (chunk size: {chunk_size})"
    )

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename without extension
    base_path = os.path.splitext(output_path)[0]
    extension = os.path.splitext(output_path)[1]

    # Calculate number of chunks needed
    total_chunks = (len(data) + chunk_size - 1) // chunk_size

    if total_chunks == 1:
        # If only one chunk needed, save with original filename
        with open(output_path, "w", encoding="utf-8") as f:
            for record in data:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        logger.info(f"Successfully saved {len(data)} records to {output_path}")
    else:
        # Save in multiple chunks
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(data))
            chunk_data = data[start_idx:end_idx]

            # Create chunk filename
            chunk_filename = f"{base_path}_chunk_{chunk_idx + 1:03d}{extension}"

            with open(chunk_filename, "w", encoding="utf-8") as f:
                for record in chunk_data:
                    json.dump(record, f, ensure_ascii=False)
                    f.write("\n")

            logger.info(
                f"Saved chunk {chunk_idx + 1}/{total_chunks}: {len(chunk_data)} records to {chunk_filename}"
            )

        logger.info(
            f"Successfully saved all {len(data)} records across {total_chunks} chunks"
        )


@click.command()
@click.option(
    "--file-metadata",
    type=click.Path(exists=True, readable=True),
    help="Path to the file-level metadata CSV file",
    # default="../envato_fonts/metadata/fonts_file_level_metadata.csv",
    required=True,
)
@click.option(
    "--zip-metadata",
    type=click.Path(exists=True, readable=True),
    help="Path to the zip-level metadata CSV file",
    # default="../envato_fonts/metadata/fonts_zip_level_metadata.csv",
    required=True,
)
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(),
    help="Output path for the processed metadata JSONL file",
    # default="data/processed_envato/metadata",
    required=True,
)
@click.option(
    "--output-log-dir",
    type=click.Path(),
    # default="data/processed_envato/",
    required=True,
)
@click.option(
    "--chunk-size",
    type=int,
    help="Number of records per chunk when saving large datasets",
    default=5000,
    show_default=True,
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def main(
    file_metadata,
    zip_metadata,
    output_dir,
    output_log_dir,
    chunk_size,
    verbose,
    overwrite,
):
    """Extract metadata of font files from Envato datasets.

    This script:
    1. Reads file-level and zip-level metadata CSV files
    2. Filters for TTF files only
    3. Merges data based on ITEM_ID
    4. Validates matching keys have same values
    5. Outputs to JSONL format in chunks
    """
    # Set logging level based on verbose flag
    global logger
    is_skip, logger = prepare_output_dir_and_logger(
        output_dir=output_dir,
        overwrite=overwrite,
        output_log_dir=output_log_dir,
    )
    if is_skip:
        exit()

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    logger.info("Starting metadata extraction process")
    logger.info(f"File metadata: {file_metadata}")
    logger.info(f"Zip metadata: {zip_metadata}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Chunk size: {chunk_size}")

    try:
        # Load metadata files
        file_metadata_df, zip_metadata_df = load_metadata_files(
            file_metadata, zip_metadata
        )

        # Filter for TTF files only
        ttf_files = filter_ttf_files(file_metadata_df)

        if len(ttf_files) == 0:
            logger.error("No TTF files found in the metadata")
            return

        # Merge metadata
        merged_data = merge_metadata(ttf_files, zip_metadata_df)

        # Save to JSONL
        save_to_jsonl(merged_data, output_dir, chunk_size)

        logger.info("Metadata extraction completed successfully")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error(
            "Please ensure the Envato metadata CSV files exist at the specified paths"
        )
        raise click.ClickException(f"File not found: {e}")
    except Exception as e:
        logger.error(f"An error occurred during metadata extraction: {e}")
        raise click.ClickException(f"Error during extraction: {e}")


if __name__ == "__main__":
    main()
