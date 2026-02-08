"""
llama-factory dataset format: https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md#alpaca-format

python -m src.svg_word_generation.build_sft_data \
    --input_svg_jsonl_dir data/processed/svg_word_generation/pangrams/metadata \
    --ouput_sft_jsonl_dir "data/processed/sft/pangrams"
"""

import json

import shutil
from pathlib import Path
from types import SimpleNamespace

import click
import tqdm

SYSTEM_PROMPT = "Design SVG code for the given text content, follow the given the font design requirements. Do not use <text> element in generated SVG, use <path> instead."

STYLE_TEMPLATE: str = """Font design requirements: {style_str}"""

CONTENT_TEMPLATE: str = """Text content: {content_str}"""


def _format_text(text):
    """
    SANS_SERIF -> Sans Serif
    """
    text = text.replace("_", " ")
    text = text.title()
    return text


def create_sft_from_row(
    row, apply_metadata_tags=True, apply_group_tags=True, verbose=False
):
    category = row["category"]
    classifications = row["classifications"]
    stroke = row["stroke"]
    style = row["style"]
    weight = row["weight"]

    # metadata tags
    if apply_metadata_tags:
        _category = []
        if category is not None:
            if isinstance(category, list):
                _category.extend([f"Category {_format_text(i)}" for i in category])
            else:
                raise ValueError(f"Unknown type for category: {type(category)}")
        if classifications is not None:
            if isinstance(classifications, list):
                _category.extend(
                    [f"Classification {_format_text(i)}" for i in classifications]
                )
            else:
                raise ValueError(
                    f"Unknown type for classifications: {type(classifications)}"
                )
        _category = list(set(_category))

        _stroke = []
        if stroke is not None:
            if isinstance(stroke, str):
                _stroke.append("Stroke " + _format_text(stroke))
            else:
                raise ValueError(f"Unknown type for stroke: {type(stroke)}")

        metadata_tags = [
            *_category,
            *_stroke,
            f"Style {style}",
            f"Weight {weight}",
        ]
    else:
        metadata_tags = []

    # human accessed expressive and typographic tags
    if apply_group_tags:
        group_tags = row["group_tags"]
        group_tags = [i.strip("/").replace("/", " ") for i in group_tags]
    else:
        group_tags = []

    # build instruction
    style_str = ", ".join(metadata_tags + group_tags)
    style_str = style_str.title()
    content_str = row["word"]

    content_str = CONTENT_TEMPLATE.format(content_str=content_str)
    if style_str:
        style_str = STYLE_TEMPLATE.format(style_str=style_str)
        instruction_str = "\n\n".join([style_str, content_str])
    else:
        instruction_str = content_str

    if verbose:
        print(f"\n========\n{instruction_str}\n========\n")

    text_svg = row["text_svg"]

    sft_row = {
        "instruction": instruction_str,
        "system": SYSTEM_PROMPT,
        "output": text_svg,
    }
    return sft_row


def get_total_rows(input_metadata_jsonl_files):
    total_rows = 0
    print(f"Reading {len(input_metadata_jsonl_files)} files to get total rows")
    for input_metadata_jsonl_file in input_metadata_jsonl_files:
        with open(input_metadata_jsonl_file, "r") as f:
            for _ in f:
                total_rows += 1
    print(f"Total rows: {total_rows}")
    return total_rows


@click.command()
@click.option(
    "--input_svg_jsonl_dir",
    default="data/processed/svg_word_generation/pangrams/metadata",
    type=str,
)
@click.option("--ouput_sft_jsonl_dir", default="data/processed/sft/pangrams", type=str)
@click.option("--chunk_size", default=1000, type=int)
@click.option("--max_dataset_size", default=None, type=int)
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite existing output."
)
@click.option("--apply_metadata_tags", is_flag=True, default=False)
@click.option("--apply_group_tags", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def main(**kwargs) -> None:
    args = SimpleNamespace(**kwargs)
    input_svg_jsonl_dir = args.input_svg_jsonl_dir
    output_sft_jsonl_dir = args.ouput_sft_jsonl_dir
    input_svg_jsonl_dir = Path(input_svg_jsonl_dir)
    output_sft_jsonl_dir = Path(output_sft_jsonl_dir)
    print(f"input_svg_jsonl_dir: {input_svg_jsonl_dir}")
    print(f"output_sft_jsonl_dir: {output_sft_jsonl_dir}")

    # prepare output dir
    overwrite = getattr(args, "overwrite", None)
    print(f"overwrite: {overwrite}. Force it to be True.")
    overwrite = True
    if output_sft_jsonl_dir.exists():
        if overwrite:
            print(
                f"Output directory {output_sft_jsonl_dir} already exists. Overwriting."
            )
            shutil.rmtree(output_sft_jsonl_dir)
            output_sft_jsonl_dir.mkdir(exist_ok=True, parents=True)
        else:
            raise FileExistsError(
                f"Output directory {output_sft_jsonl_dir} already exists. Use --overwrite to overwrite."
            )
    else:
        print(f"Output directory {output_sft_jsonl_dir} does not exist. Creating.")
        output_sft_jsonl_dir.mkdir(exist_ok=True, parents=True)

    # get input jsonl files
    input_svg_jsonl_files = list(input_svg_jsonl_dir.glob("*.jsonl"))
    print(f"Found {len(input_svg_jsonl_files)} files in {input_svg_jsonl_dir}")

    # write to sft jsonl files
    chunk_size = args.chunk_size
    chunk_size: int
    max_dataset_size = getattr(args, "max_dataset_size", None)
    current_size = 0
    chunk_idx = 0

    apply_metadata_tags = args.apply_metadata_tags
    apply_group_tags = args.apply_group_tags
    verbose = args.verbose
    sft_data_list = []
    total_rows = get_total_rows(input_svg_jsonl_files)
    pbar = tqdm.tqdm(total=total_rows, desc="Processing jsonl files")
    for input_svg_jsonl_file in input_svg_jsonl_files:
        with open(input_svg_jsonl_file, "r") as f:
            for line in f:
                if max_dataset_size is not None and current_size >= max_dataset_size:
                    break

                data = json.loads(line)
                sft_data = create_sft_from_row(
                    data,
                    apply_metadata_tags=apply_metadata_tags,
                    apply_group_tags=apply_group_tags,
                    verbose=verbose,
                )
                sft_data_list.append(sft_data)
                current_size += 1
                pbar.update(1)

                if len(sft_data_list) >= chunk_size:
                    output_sft_jsonl_path = (
                        output_sft_jsonl_dir / f"{chunk_idx:05d}.jsonl"
                    )
                    with open(output_sft_jsonl_path, "a", buffering=8192 * 16) as f:
                        for sft_data in sft_data_list:
                            f.write(json.dumps(sft_data) + "\n")
                    sft_data_list.clear()
                    chunk_idx += 1

    if len(sft_data_list) > 0:
        output_sft_jsonl_path = output_sft_jsonl_dir / f"{chunk_idx:05d}.jsonl"
        with open(output_sft_jsonl_path, "a", buffering=8192 * 16) as f:
            for sft_data in sft_data_list:
                f.write(json.dumps(sft_data) + "\n")
        sft_data_list.clear()
        chunk_idx += 1


if __name__ == "__main__":
    main()
