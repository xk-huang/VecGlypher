#!/usr/bin/env python3
"""
Extract fonts from zip files using metadata.

This script extracts font files from zip archives based on metadata information.
It uses parallel processing to handle multiple files efficiently and includes
error handling for broken zip files.

python -m src.svg_glyph_gen_v2_envato.extract_zip_fonts
"""

import hashlib
import json
import logging
import zipfile
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from shutil import rmtree
from typing import Any, Dict, List, Optional, Tuple

import click
from tqdm import tqdm

from ..svg_glyph_gen_v2.utils import prepare_output_dir_and_logger

# Configure logging
logger = logging.getLogger(__name__)


def load_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON: {e}")
    return data


def load_metadata(metadata_dir: Path) -> List[Dict[str, Any]]:
    """Load metadata from JSONL files in the metadata directory."""
    metadata = []

    if not metadata_dir.exists():
        logger.error(f"Metadata directory does not exist: {metadata_dir}")
        return metadata

    for json_file in metadata_dir.glob("*.jsonl"):
        try:
            metadata.extend(load_jsonl(json_file))
            logger.info(f"Loaded metadata from {json_file}")
        except Exception as e:
            logger.error(f"Error loading metadata from {json_file}: {e}")

    logger.info(f"Total metadata entries loaded: {len(metadata)}")
    return metadata


def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def extract_font_from_zip(
    item_id: str,
    path_in_zip: str,
    zip_font_dir: Path,
    output_dir: Path,
    expected_md5: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Extract a single font file from a zip archive and verify MD5 hash.

    Returns:
        Tuple of (success: bool, message: str)
    """
    # Construct zip file path
    zip_file_path = zip_font_dir / item_id / f"{item_id}.zip"

    if not zip_file_path.exists():
        return False, f"Zip file not found: {zip_file_path}"

    try:
        # Create output directory structure
        output_path = output_dir / item_id / path_in_zip
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract font file from zip
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            try:
                # Check if the file exists in the zip
                if path_in_zip not in zip_ref.namelist():
                    return False, f"Font file not found in zip: {path_in_zip}"

                # Extract the specific file
                with zip_ref.open(path_in_zip) as source:
                    with open(output_path, "wb") as target:
                        target.write(source.read())

                # Verify MD5 hash if provided
                if expected_md5:
                    actual_md5 = calculate_md5(output_path)
                    if actual_md5.lower() != expected_md5.lower():
                        return (
                            False,
                            f"MD5 mismatch for {path_in_zip}: expected {expected_md5}, got {actual_md5}",
                        )
                    return (
                        True,
                        f"Successfully extracted and verified {path_in_zip} (MD5: {actual_md5})",
                    )
                else:
                    # remove the file if no MD5 is provided
                    output_path.unlink()
                    return (
                        False,
                        f"Successfully extracted {path_in_zip} to {output_path} (no MD5 verification)",
                    )

            except KeyError:
                return False, f"Font file not found in zip: {path_in_zip}"
            except Exception as e:
                return False, f"Error extracting {path_in_zip}: {e}"

    except zipfile.BadZipFile:
        return False, f"Corrupted zip file: {zip_file_path}"
    except Exception as e:
        return False, f"Unexpected error processing {item_id}: {e}"


def process_metadata_entry(
    entry: Dict[str, Any], zip_font_dir: Path, output_dir: Path
) -> Tuple[str, bool, str]:
    """
    Process a single metadata entry.

    Returns:
        Tuple of (item_id: str, success: bool, message: str)
    """
    item_id = entry.get("ITEM_ID")
    path_in_zip = entry.get("PATH_IN_ZIP")
    expected_md5 = entry.get("MD5")

    if not item_id:
        return "unknown", False, "Missing ITEM_ID in metadata entry"

    if not path_in_zip:
        return item_id, False, "Missing PATH_IN_ZIP in metadata entry"

    success, message = extract_font_from_zip(
        item_id, path_in_zip, zip_font_dir, output_dir, expected_md5
    )

    return item_id, success, message


@click.command()
@click.option(
    "--zip-font-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    # default="../envato_fonts/zip_fonts",
    help="Directory containing zip font files",
    required=True,
)
@click.option(
    "--metadata-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    # default="data/processed_envato/metadata",
    help="Directory containing metadata JSON files",
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    # default="../envato_fonts/fonts",
    help="Output directory for extracted fonts",
    required=True,
)
@click.option(
    "--output-log-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    # default="data/processed_envato/",
    help="Output directory for extracted fonts",
    required=True,
)
@click.option("--workers", type=int, default=20, help="Number of parallel workers")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level",
)
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def main(
    zip_font_dir: Path,
    metadata_dir: Path,
    output_dir: Path,
    output_log_dir: Path,
    workers: int,
    log_level: str,
    debug: bool,
    overwrite: bool,
):
    """Extract fonts from zip files using metadata information."""

    global logger
    is_skip, logger = prepare_output_dir_and_logger(
        output_dir=output_dir,
        overwrite=overwrite,
        output_log_dir=output_log_dir,
    )
    if is_skip:
        exit()

    # Set logging level
    logger.setLevel(getattr(logging, log_level))

    logger.info("Starting font extraction process")
    logger.info(f"Zip font directory: {zip_font_dir}")
    logger.info(f"Metadata directory: {metadata_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of workers: {workers}")

    # Create output directory
    if output_dir.exists() and overwrite:
        rmtree(output_dir)
        logger.warning(f"Output directory already exists. Overwriting: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata = load_metadata(metadata_dir)
    if not metadata:
        logger.error("No metadata loaded. Exiting.")
        return

    # local test
    if debug:
        entry = metadata[0]
        item_id, success, message = process_metadata_entry(
            entry, zip_font_dir, output_dir
        )
        print(f"item_id: {item_id}, success: {success}, message: {message}")
        return

    # Process entries in parallel
    successful_extractions = 0
    failed_extractions = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_entry = {
            executor.submit(
                process_metadata_entry, entry, zip_font_dir, output_dir
            ): entry
            for entry in metadata
        }

        # Process completed tasks
        for future in tqdm(as_completed(future_to_entry), total=len(future_to_entry)):
            try:
                item_id, success, message = future.result()

                if success:
                    successful_extractions += 1
                    logger.debug(f"✓ {item_id}: {message}")
                else:
                    failed_extractions += 1
                    logger.warning(f"✗ {item_id}: {message}")

            except Exception as e:
                failed_extractions += 1
                logger.error(f"✗ Error processing entry: {e}")

    # Summary
    total_entries = len(metadata)
    logger.info("=" * 60)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total entries processed: {total_entries}")
    logger.info(f"Successful extractions: {successful_extractions}")
    logger.info(f"Failed extractions: {failed_extractions}")
    logger.info(f"Success rate: {successful_extractions/total_entries*100:.1f}%")
    logger.info("=" * 60)

    if failed_extractions > 0:
        logger.warning(
            f"Check the log file for details on {failed_extractions} failed extractions"
        )


if __name__ == "__main__":
    main()
