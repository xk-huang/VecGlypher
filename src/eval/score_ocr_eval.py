#!/usr/bin/env python3
"""
Process JSONL files under a directory (recursively) and write results to score.jsonl.

Usage:
  python -m src.eval.score_ocr_eval <input_dir> [<output_dir>]

- Finds all *.jsonl under input_dir (including subdirectories).
- Streams records line-by-line, so it works on large files.
- Applies `process_record(record)` to each parsed dict.
- Writes all processed records to a single file: <output_dir>/score.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import click
import tqdm

from ..svg_glyph_gen_v2.build_sft_data_v2 import WORD_SEP
from ..svg_glyph_gen_v2.utils import prepare_output_dir_and_logger

SCORE_FILE_NAME = "score.jsonl"

# NOTE: the below characters are used in word. e.g. "fine-tuning", "our's"
# They are not intended to be used as a single character.
INVALID_CHAR_SET = "-'"
INVALID_CHAR_SET = set(list(INVALID_CHAR_SET))


def process_record(
    record: Dict[str, Any], use_case: bool, remove_whitespace: bool
) -> Optional[Dict[str, Any]]:
    """
    Dummy transformation. Modify this function to implement your logic.

    Return:
        - A dict to write to the output JSONL
        - None to skip writing this record
    """
    # Example: pass through and add a dummy score
    predict = record["predict"]
    metadata = json.loads(record["metadata"])
    gt = metadata["content_str"]

    if gt in INVALID_CHAR_SET:
        return None

    predict = predict.strip()
    gt = gt.strip()

    # replace word separator "<|SEP|>"
    predict = predict.replace(WORD_SEP, "")
    gt = gt.replace(WORD_SEP, "")

    if remove_whitespace:
        predict = "".join(predict.split())

    if use_case is False:
        predict = predict.lower()
        gt = gt.lower()

    record["parsed_predict"] = predict
    record["parsed_gt"] = gt
    record["score"] = predict == gt
    return record


def iter_jsonl_files(root: Path) -> Iterable[Path]:
    """Yield all *.jsonl files under root recursively."""
    for i in root.rglob("*.jsonl"):
        if i.is_file() and i.name != SCORE_FILE_NAME:
            yield i


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument(
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default="outputs/debug/infer_ocr_eval",
)
@click.argument(
    "output_dir", required=True, type=click.Path(file_okay=False, path_type=Path)
)
@click.option("--use_case/--no_use_case", default=True, help="Case sensitive")
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--remove_whitespace/--no_remove_whitespace", default=False)
@click.option("--skip_number/--no_skip_number", default=False)
def main(
    input_dir: Path,
    output_dir: Optional[Path],
    use_case: bool,
    overwrite: bool,
    remove_whitespace: bool,
    skip_number: bool,
) -> None:
    """
    INPUT_DIR: Directory to search for *.jsonl (recursively).
    OUTPUT_DIR: Optional output directory. If not provided, uses INPUT_DIR.
    """
    output_dir = output_dir or input_dir
    should_skip, logger = prepare_output_dir_and_logger(
        output_dir=output_dir,
        overwrite=overwrite,
    )
    if should_skip:
        exit()
    output_dir = Path(output_dir)

    out_path = output_dir / SCORE_FILE_NAME

    jsonl_files = sorted(iter_jsonl_files(input_dir))
    logger.info(f"Found {len(jsonl_files)} JSONL file(s) under: {input_dir}")

    total_in = 0
    total_out = 0
    total_err = 0

    if skip_number:
        logger.info(f"skip numbers, add 0-9 to INVALID_CHAR_SET")
        for i in range(10):
            INVALID_CHAR_SET.add(str(i))

    total_correct = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for fpath in jsonl_files:
            logger.info(f"- Processing: {fpath}")
            with fpath.open("r", encoding="utf-8") as fin:
                for lineno, line in enumerate(tqdm.tqdm(fin), start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("JSONL line is not a JSON object (dict).")
                        total_in += 1
                        processed = process_record(record, use_case, remove_whitespace)
                        if processed is not None:
                            fout.write(json.dumps(processed, ensure_ascii=False) + "\n")
                            total_out += 1
                            total_correct += processed["score"]
                    except Exception as e:
                        total_err += 1
                        logger.info(
                            f"  ! Error in {fpath}:{lineno}: {e}",
                            err=True,
                        )

    logger.info(
        f"Done. Input records: {total_in} | Written: {total_out} | Errors: {total_err}\n"
        f"Output: {out_path}"
    )
    out_stats_path = output_dir / f"score_stats-use_case_{use_case}.json"
    with open(out_stats_path, "w") as f:
        stats = {
            "total_in": total_in,
            "total_out": total_out,
            "total_err": total_err,
            "use_case": use_case,
            "total_correct": total_correct,
            "accuracy": total_correct / total_out,
        }
        json.dump(stats, f, ensure_ascii=False, indent=4)
    logger.info(f"Stats: {stats}")
    logger.info(f"Save Stats: {out_stats_path}")


if __name__ == "__main__":
    main()
