"""
python -m src.svg_glyph_gen_v2.gather_content
"""

import logging
import runpy
import shutil
import string
import subprocess
from pathlib import Path

import click
import pandas as pd

from .filter_invalid_fonts import VALID_CHAR_SET

from .utils import setup_logger

logger = logging.getLogger(__name__)


def run_command(command):
    logger.info(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError("Error running command:", result.stderr)
    logger.info(f"command stdout: {result.stdout}")


@click.command()
@click.argument(
    "output_dir",
    type=click.Path(),
    # default="data/processed/content",
    required=True,
)
def main(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    global logger
    logger = setup_logger(output_dir)

    prepare_alphanumeric_chars(output_dir)
    diacritics_path = Path(__file__).resolve().parent / "diacritics.py"
    diacritics = runpy.run_path(diacritics_path)["latin_diacritics"]
    output_path = output_dir / "diacritics.txt"
    with open(output_path, "w") as f:
        for char in diacritics:
            f.write(char + "\n")
    logger.info(f"Number of diacritics: {len(diacritics)}")
    logger.info(f"Written to {output_path}")

    # download_oxford_3000_5000(output_dir)
    # prepare_oxford_3000_5000(output_dir)

    # remove_alphanumeric_in_oxford_3000_5000(output_dir)

    # merge_alphanumeric_and_oxford_3000_5000(output_dir)


def merge_alphanumeric_and_oxford_3000_5000(output_dir):
    alphanumeric_name = "alphanumeric.txt"
    oxford_names = ["oxford-3000.txt", "oxford-5000.txt"]
    for oxford_name in oxford_names:
        merge_alphanumeric_and_oxford(output_dir, alphanumeric_name, oxford_name)


def merge_alphanumeric_and_oxford(output_dir, alphanumeric_name, oxford_name):
    alphanumeric_path = output_dir / alphanumeric_name
    o3k_path = output_dir / oxford_name

    with open(alphanumeric_path, "r") as f:
        alphanumeric_chars = set(f.read().splitlines())

    with open(o3k_path, "r") as f:
        oxford_words = set(f.read().splitlines())

    oxford_words = oxford_words.union(alphanumeric_chars)
    output_path = output_dir / f"{alphanumeric_path.stem}-{o3k_path.stem}.txt"
    with open(output_path, "w") as f:
        for word in sorted(oxford_words):
            f.write(word + "\n")
    logger.info(f"Merge {alphanumeric_name} and {oxford_name}: {len(oxford_words)}")
    logger.info(f"Written to {output_path}")


def remove_alphanumeric_in_oxford_3000_5000(output_dir):
    alphanumeric_name = "alphanumeric.txt"
    oxford_names = ["oxford-3000.txt", "oxford-5000.txt"]
    for oxford_name in oxford_names:
        remove_alphanumeric_in_oxford(output_dir, alphanumeric_name, oxford_name)


def remove_alphanumeric_in_oxford(output_dir, alphanumeric_name, oxford_name):
    alphanumeric_path = output_dir / alphanumeric_name
    o3k_path = output_dir / oxford_name

    with open(alphanumeric_path, "r") as f:
        alphanumeric_chars = set(f.read().splitlines())

    with open(o3k_path, "r") as f:
        oxford_words = set(f.read().splitlines())

    len_before = len(oxford_words)
    oxford_words = oxford_words - alphanumeric_chars
    len_after = len(oxford_words)
    with open(o3k_path, "w") as f:
        for word in sorted(oxford_words):
            f.write(word + "\n")
    logger.info(
        f"Deduplicate {alphanumeric_name} from {oxford_name}: {len_before} -> {len_after}"
    )


def prepare_alphanumeric_chars(output_dir):
    # Write to file
    output_path = output_dir / "alphanumeric.txt"
    with open(output_path, "w") as f:
        for char in VALID_CHAR_SET:
            f.write(char + "\n")

    logger.info(f"Number of words of alphanumeric: {len(VALID_CHAR_SET)}")
    logger.info(f"Written to {output_path}")


def prepare_oxford_3000_5000(output_dir):

    for csv_name in ["oxford-3000.csv", "oxford-5000.csv"]:
        csv_path = output_dir / csv_name
        txt_path = csv_path.with_suffix(".txt")
        sort_dedup_save_word(csv_path, txt_path)
        logger.info(f"Written {csv_path} to {txt_path}")

        if csv_path.exists():
            csv_path.unlink()
            logger.info(f"Removed {csv_path}")


def sort_dedup_save_word(input_csv_path, output_txt_path):
    df = pd.read_csv(input_csv_path)
    word = df["word"].sort_values()
    word = word.drop_duplicates()
    with open(output_txt_path, "w") as f:
        for w in word:
            f.write(w + "\n")
    logger.info(f"Deduplication. Number of words: {len(df)} -> {len(word)}")


def download_oxford_3000_5000(output_dir):
    # downlod repo and unzip
    repo_hash = "8ffbf197309767658a92553f74f6993e7e89e322"
    command = [
        "wget",
        f"https://github.com/Berehulia/Oxford-3000-5000/archive/{repo_hash}.zip",
        "-O",
        str(output_dir / "Oxford-3000-5000.zip"),
    ]
    run_command(command)

    # unzip
    command = [
        "unzip",
        "-o",
        str(output_dir / "Oxford-3000-5000.zip"),
        "-d",
        str(output_dir),
    ]
    run_command(command)

    # Move contents from the nested directory up one level
    extracted_dir = output_dir / f"Oxford-3000-5000-{repo_hash}"
    if extracted_dir.exists():
        # Move all contents from extracted_dir to output_dir
        for item in extracted_dir.iterdir():
            dest = output_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))

        # Remove the now-empty extracted directory
        extracted_dir.rmdir()
        logger.info(f"Moved contents from {extracted_dir.name} to {output_dir}")

    # Clean up the zip file
    zip_file = output_dir / "Oxford-3000-5000.zip"
    if zip_file.exists():
        zip_file.unlink()
        logger.info(f"Removed zip file: {zip_file}")


if __name__ == "__main__":
    main()
