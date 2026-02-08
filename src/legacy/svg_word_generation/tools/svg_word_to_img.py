"""
Visulize SVG by rendering to images

Inputs: font-word jsonl
outputs: font-word pdf

python src/svg_word_generation/tools/svg_word_to_img.py \
    --input_dir data/processed/svg_word_generation/pangrams-skia
"""

import json
from concurrent.futures import as_completed, ProcessPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import cairosvg
import click
import tqdm


OUTPUT_TYPES = ["pdf", "png"]
WRITE_FUNC_MAPPING = {
    "pdf": cairosvg.svg2pdf,
    "png": cairosvg.svg2png,
}
OUTPUT_HEIGHT = 50


def process_svg_data(data, output_dir, output_type):
    """Process a single SVG data entry."""
    text_svg = data["text_svg"]
    output_hash = data["output_hash"]
    output_path = output_dir / f"{output_hash}.{output_type}"
    WRITE_FUNC_MAPPING[output_type](
        bytestring=text_svg.encode("utf-8"),
        write_to=str(output_path),
        output_height=OUTPUT_HEIGHT,
    )
    return output_hash


def process_jsonl_file(input_jsonl_path, output_dir, output_type, max_workers=None):
    """Process a single JSONL file with parallel SVG rendering."""
    data_list = []
    with open(input_jsonl_path, "r") as f:
        for line in f.readlines():
            data_list.append(json.loads(line))

    # Create a partial function with fixed arguments
    process_func = partial(
        process_svg_data, output_dir=output_dir, output_type=output_type
    )

    processed_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_data = {
            executor.submit(process_func, data): data for data in data_list
        }

        # Process completed tasks with progress bar
        for future in tqdm.tqdm(
            as_completed(future_to_data),
            total=len(data_list),
            desc=f"rendering {input_jsonl_path.name}",
        ):
            try:
                future.result()
                processed_count += 1
            except Exception as exc:
                data = future_to_data[future]
                print(
                    f'SVG processing failed for hash {data.get("output_hash", "unknown")}: {exc}'
                )

    return processed_count


@click.command()
@click.option(
    "--input_dir",
    type=click.Path(exists=True, file_okay=False),
    default="data/processed/svg_word_generation/pangrams",
)
@click.option("--output_type", type=click.Choice(OUTPUT_TYPES), default="png")
@click.option(
    "--max_workers", type=int, default=None, help="Maximum number of worker processes"
)
@click.option(
    "--file_parallelism",
    is_flag=True,
    help="Enable parallelism at file level instead of within files",
)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    output_type = getattr(args, "output_type", None)
    if output_type not in OUTPUT_TYPES:
        raise ValueError(f"Unknown output type: {output_type}")

    max_workers = getattr(args, "max_workers", None)
    file_parallelism = getattr(args, "file_parallelism", False)

    input_dir = getattr(args, "input_dir", None)
    input_dir = Path(input_dir)
    input_jsonl_dir = input_dir / "metadata"

    output_dir = input_dir / output_type
    output_dir.mkdir(exist_ok=True, parents=True)

    input_jsonl_list = list(input_jsonl_dir.glob("*.jsonl"))

    print(f"Input dir: {input_jsonl_dir}")
    print(f"Found {len(input_jsonl_list)} jsonl files")
    print(f"Max workers: {max_workers}")
    print(f"File-level parallelism: {file_parallelism}")

    if file_parallelism:
        # Parallelize at the file level
        process_file_func = partial(
            process_jsonl_file,
            output_dir=output_dir,
            output_type=output_type,
            max_workers=1,  # Use single worker per file when parallelizing files
        )

        total_processed = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_file_func, jsonl_path): jsonl_path
                for jsonl_path in input_jsonl_list
            }

            for future in tqdm.tqdm(
                as_completed(future_to_file),
                total=len(input_jsonl_list),
                desc="processing jsonl files",
            ):
                try:
                    processed_count = future.result()
                    total_processed += processed_count
                except Exception as exc:
                    jsonl_path = future_to_file[future]
                    print(f"File processing failed for {jsonl_path.name}: {exc}")

        print(f"Total processed: {total_processed}")
    else:
        # Parallelize within each file (default behavior)
        total_processed = 0
        for input_jsonl_path in tqdm.tqdm(
            input_jsonl_list, desc="processing jsonl files"
        ):
            processed_count = process_jsonl_file(
                input_jsonl_path, output_dir, output_type, max_workers
            )
            total_processed += processed_count

        print(f"Total processed: {total_processed}")


if __name__ == "__main__":
    main()
