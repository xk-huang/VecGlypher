#!/usr/bin/env python3
"""
Convert Envato metadata format to Google Font metadata format.

This script converts metadata from Envato format to Google Font format,
processing JSONL files in chunks and saving the results.

python -m src.svg_glyph_gen_v2_envato.convert_to_gfont_format
"""

import json
import logging
from pathlib import Path

from shutil import rmtree
from typing import Any, Dict

import click
from tqdm import tqdm

from ..svg_glyph_gen_v2.utils import prepare_output_dir_and_logger

# Set up logger
logger = logging.getLogger(__name__)


def convert_envato_to_gfont_format(envato_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single Envato metadata record to Google Font format.

    Args:
        envato_record: Dictionary containing Envato metadata

    Returns:
        Dictionary in Google Font metadata format
    """
    # Extract filename from PATH_IN_ZIP
    filename = envato_record.get("PATH_IN_ZIP", None)

    name = envato_record.get("TITLE", None)
    source = envato_record.get("BRAND", None)

    category = envato_record.get("CATEGORY", None)
    if category is not None:
        category = json.loads(category)
    tags = envato_record.get("TAGS", None)
    if tags is not None:
        tags = json.loads(tags)
    spacings = envato_record.get("SPACINGS", None)
    if spacings is not None:
        spacings = json.loads(spacings)

    font_family_dir_name = envato_record.get("ITEM_ID", None)

    # Create the converted record with the filename conversion
    converted_record = {
        "filename": filename,
        "name": name,
        "source": source,
        "category": category,
        "tags": tags,
        "spacings": spacings,
        "font_family_dir_name": font_family_dir_name,
    }

    return converted_record


def process_metadata_chunk(
    input_file: Path, output_file: Path, show_progress: bool = True
) -> int:
    """
    Process a single metadata chunk file.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        show_progress: Whether to show progress bar for individual file processing

    Returns:
        Number of records processed
    """
    records_processed = 0

    # Count total lines for progress bar
    total_lines = 0
    if show_progress:
        with open(input_file, "r", encoding="utf-8") as f:
            total_lines = sum(1 for line in f if line.strip())

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:

        # Create progress bar for individual file processing
        line_iterator = infile
        if show_progress and total_lines > 0:
            line_iterator = tqdm(
                infile,
                total=total_lines,
                desc=f"Processing {input_file.name}",
                unit="records",
                leave=False,
            )

        for line in line_iterator:
            line = line.strip()
            if not line:
                continue

            try:
                envato_record = json.loads(line)
                converted_record = convert_envato_to_gfont_format(envato_record)

                # Write converted record to output file
                json.dump(converted_record, outfile, ensure_ascii=False)
                outfile.write("\n")

                records_processed += 1

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON in {input_file}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing record in {input_file}: {e}")
                continue

    return records_processed


@click.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    # default="/home/vecglypher/codes/svg_glyph_llm_data/processed_envato/metadata",
    help="Input directory containing metadata_chunk_*.jsonl files",
    required=True,
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    # default="/home/vecglypher/codes/svg_glyph_llm_data/processed_envato/metadata_in_gfont",
    help="Output directory for converted metadata files",
    required=True,
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress bars",
)
@click.option(
    "--output-log-dir",
    "-l",
    type=click.Path(path_type=Path),
    help="Optional log file path. If provided, logs will be written to this file",
    # default="data/processed_envato",
    required=True,
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files")
def main(
    input_dir: Path,
    output_dir: Path,
    no_progress: bool,
    output_log_dir: Path,
    overwrite: bool,
):
    """
    Convert Envato metadata format to Google Font metadata format.

    This script processes all metadata_chunk_*.jsonl files in the input directory
    and converts them from Envato format to Google Font format, saving the results
    in the output directory with 'converted_' prefix.
    """
    # Set up logging
    global logger
    is_skip, logger = prepare_output_dir_and_logger(
        output_dir=output_dir,
        overwrite=overwrite,
        output_log_dir=output_log_dir,
    )
    if is_skip:
        exit()

    show_progress = not no_progress

    # Create output directory if it doesn't exist
    if output_dir.exists() and overwrite:
        logger.warning(f"Output directory already exists: {output_dir}")
        click.echo(f"Output directory already exists: {output_dir}", err=True)
        rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created output directory: {output_dir}")

    # Find all metadata chunk files
    input_files = sorted(input_dir.glob("metadata_chunk_*.jsonl"))

    if not input_files:
        logger.error(f"No metadata chunk files found in {input_dir}")
        click.echo(f"No metadata chunk files found in {input_dir}", err=True)
        return

    logger.info(f"Found {len(input_files)} metadata chunk files to process")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Also show basic info via click for immediate user feedback
    click.echo(f"Found {len(input_files)} metadata chunk files to process")

    total_records = 0

    # Create progress bar for overall file processing
    file_iterator = input_files
    if show_progress:
        file_iterator = tqdm(
            input_files, desc="Processing files", unit="files", position=0
        )

    # Process each chunk file
    for input_file in file_iterator:
        output_file = output_dir / f"converted_{input_file.name}"

        logger.debug(f"Processing {input_file.name}...")

        try:
            records_processed = process_metadata_chunk(
                input_file, output_file, show_progress=show_progress
            )
            total_records += records_processed

            logger.info(
                f"Processed {records_processed} records from {input_file.name} -> {output_file.name}"
            )

        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}")
            continue

    logger.info("Conversion complete!")
    logger.info(f"Total records processed: {total_records}")
    logger.info(f"Output files saved to: {output_dir}")

    # Also show completion info via click for immediate user feedback
    click.echo("\nConversion complete!")
    click.echo(f"Total records processed: {total_records}")
    click.echo(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()
