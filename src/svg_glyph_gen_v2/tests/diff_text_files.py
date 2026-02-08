"""
python src/svg_glyph_gen_v2/tests/diff_oxford.py
"""

from pathlib import Path

import click
from transformers.models.llama.tokenization_llama import B_INST


@click.command()
@click.argument(
    "file_a",
    type=click.Path(exists=True),
    default="data/processed/content/oxford-3000.txt",
)
@click.argument(
    "file_b",
    type=click.Path(exists=True),
    default="data/processed/content/oxford-5000.txt",
)
def main(file_a, file_b):
    text_a = Path(file_a).read_text().strip().split("\n")
    text_b = Path(file_b).read_text().strip().split("\n")

    a_diff_b = set(text_a) - set(text_b)
    b_diff_a = set(text_b) - set(text_a)

    print(f"{file_a} diff {file_b}: {len(a_diff_b)}")
    print(f"{file_b} diff {file_a}: {len(b_diff_a)}")


if __name__ == "__main__":
    main()
