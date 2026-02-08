"""
python -m src.serve.decode_to_svg <input_dir> [<output_dir>]

Parallel version with tqdm progress bars.
"""

import json
import os
from concurrent.futures import process, ProcessPoolExecutor
from pathlib import Path

import click
from tqdm.auto import tqdm

from ..svg_glyph_gen_v2.svg_simplifier import SVGSimplifier
from ..svg_glyph_gen_v2.utils import (
    load_jsonl,
    prepare_output_dir_and_logger,
    write_jsonl,
)

# Per-process singleton, initialized once in each worker
_svg = None


def _init_worker():
    """Initializer run once per worker process."""
    global _svg


_svg = SVGSimplifier()


def _process_line(line: str, logger) -> str:
    """Decode a single JSONL line using the per-process SVGSimplifier."""
    if not line.strip():
        return ""
    data = json.loads(line)

    if "predict" in data:
        predict = data["predict"]
        decoded_predict = _svg.decode(predict, logger=logger)
        data["predict"] = decoded_predict

    if "label" in data:
        label = data["label"]
        decoded_label = _svg.decode(label, logger=logger)
        data["label"] = decoded_label

    if "output" in data:
        output = data["output"]
        decoded_output = _svg.decode(output, logger=logger)
        data["output"] = decoded_output

    return json.dumps(data, ensure_ascii=False)


def process_batch(data, logger):
    result = []
    for line in data:
        result.append(_process_line(line, logger))
    return result


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True), default="outputs/debug/raw_outputs"
)
@click.argument("output_dir", type=str, default=None, required=False)
@click.option(
    "--num_workers",
    "-w",
    type=int,
    default=20,
    show_default=True,
    help="Number of parallel worker processes.",
)
@click.option(
    "--batch_size",
    type=int,
    default=512,
    show_default=True,
    help="batch size for each worker.",
)
@click.option("--overwrite", is_flag=True, default=False)
def main(input_dir, output_dir, num_workers, batch_size, overwrite):
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir.with_name(f"{input_dir.stem}_decoded")
    else:
        output_dir = Path(output_dir)
    # prepare output dir and logger
    should_skip, logger = prepare_output_dir_and_logger(
        output_dir=output_dir,
        overwrite=overwrite,
    )
    if should_skip:
        exit()

    data = load_jsonl(input_dir, logger=logger, decode_jsonl=False)

    if num_workers == 0:
        _init_worker()
        results = []
        for start_idx in tqdm(range(0, len(data), batch_size), desc="processing"):
            end_idx = min(start_idx + batch_size, len(data))
            results.extend(process_batch(data[start_idx:end_idx], logger=logger))
        write_jsonl(
            results, output_dir / "decoded.jsonl", logger=logger, encode_jsonl=False
        )
        return

    with ProcessPoolExecutor(num_workers, initializer=_init_worker) as executor:
        futures = []
        for start_idx in tqdm(range(0, len(data), batch_size), desc="submitting jobs"):
            end_idx = min(start_idx + batch_size, len(data))
            futures.append(
                executor.submit(
                    process_batch,
                    data[start_idx:end_idx],
                    logger=logger,
                )
            )
        results = []
        for future in tqdm(futures, desc="collecting results"):
            results.extend(future.result())
    write_jsonl(
        results, output_dir / "decoded.jsonl", logger=logger, encode_jsonl=False
    )


if __name__ == "__main__":
    main()
