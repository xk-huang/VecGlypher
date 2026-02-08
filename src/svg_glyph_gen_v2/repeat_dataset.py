"""
python -m src.svg_glyph_gen_v2.repeat_dataset \
    data/processed_envato/filtered_sft/250903-alphanumeric/train_font_family \
    data/processed_envato/filtered_sft/250903-alphanumeric/train_font_family-5x \
    --num_repeats 5

python src/svg_glyph_gen_v2/build_dataset_info.py data/processed_envato/filtered_sft/250903-alphanumeric/
python src/svg_glyph_gen_v2/stat_sft_data.py data/processed_envato/filtered_sft/250903-alphanumeric
"""

import json
from pathlib import Path

import click

from .utils import load_jsonl, write_jsonl


@click.command()
@click.argument("input_jsonl_dir", required=True)
@click.argument("output_jsonl_dir", required=True)
@click.option("--num_repeats", required=True, type=int, help="Number of repeats")
def main(input_jsonl_dir, output_jsonl_dir, num_repeats):
    input_jsonl_dir = Path(input_jsonl_dir)
    if not input_jsonl_dir.exists():
        raise FileNotFoundError(f"{input_jsonl_dir} does not exist.")

    output_jsonl_dir = Path(output_jsonl_dir)
    if output_jsonl_dir.exists():
        raise FileExistsError(
            f"{output_jsonl_dir} already exists. Make sure to delete it first."
        )
    print(f"Creating {output_jsonl_dir}...")
    print(f"Repeating {input_jsonl_dir} for {num_repeats} times...")

    input_data = load_jsonl(input_jsonl_dir)
    repeat_input_data = input_data * num_repeats
    print(f"Repeating {len(input_data)} records to {len(repeat_input_data)} records.")

    output_jsonl_path = output_jsonl_dir / "data.jsonl"
    write_jsonl(repeat_input_data, output_jsonl_path)
    print(f"Saved to {output_jsonl_path}.")


if __name__ == "__main__":
    main()
