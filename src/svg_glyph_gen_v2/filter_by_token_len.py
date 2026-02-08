#!/usr/bin/env python3
import json
import shutil
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import click
from transformers import AutoTokenizer


def _prepare_additional_tokens(add_special_tokens: str) -> List[str]:
    if not add_special_tokens:
        return []
    # Split by comma and drop empty/whitespace-only entries
    return [t for t in (tok.strip() for tok in add_special_tokens.split(",")) if t]


def _load_tokenizer(name_or_path: str, additional_special_tokens: List[str]):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if additional_special_tokens:
        tok.add_special_tokens(
            {"additional_special_tokens": additional_special_tokens},
            replace_additional_special_tokens=False,
        )
    return tok


def _process_single_file(
    file_path_str: str,
    output_dir_str: str,
    tokenizer_name_or_path: str,
    additional_special_tokens: List[str],
    max_token_len: int,
) -> Dict:
    """
    Runs in a separate process. Loads its own tokenizer instance.
    Returns stats dict.
    """
    in_path = Path(file_path_str)
    out_dir = Path(output_dir_str)
    out_path = out_dir / in_path.name

    tokenizer = _load_tokenizer(tokenizer_name_or_path, additional_special_tokens)

    n_in = 0
    n_out = 0
    n_removed = 0
    n_invalid = 0

    # Stream read/write to avoid high memory usage
    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            n_in += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_invalid += 1
                continue

            output_val = obj.get("output", None)
            if not isinstance(output_val, str):
                # Treat non-string/missing output as invalid row; drop it.
                n_invalid += 1
                continue

            # Compute token length (content only)
            # Note: using add_special_tokens=False so length reflects text itself
            token_ids = tokenizer.encode(output_val, add_special_tokens=False)
            if len(token_ids) > max_token_len:
                n_removed += 1
                continue

            # Keep this row
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1

    return {
        "file": in_path.name,
        "input_rows": n_in,
        "output_rows": n_out,
        "removed_rows": n_removed,
        "invalid_rows": n_invalid,
        "output_path": str(out_path),
    }


DEFAULT_MODEL = (
    "saves/Qwen/Qwen3-4B"
)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument(
    "input_jsonl_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("tokenizer_name_or_path", type=str, default=DEFAULT_MODEL)
@click.option("--max_token_len", default=6000, show_default=True, type=int)
@click.option("--num_worker", default=20, show_default=True, type=int)
@click.option(
    "--add_special_tokens",
    default="",
    show_default=True,
    type=str,
    help='Comma-separated list, e.g. "<|image|>,<|eos_reason|>"',
)
def main(
    input_jsonl_dir: Path,
    output_dir: Path,
    tokenizer_name_or_path: str,
    max_token_len: int,
    num_worker: int,
    add_special_tokens: str,
):
    """
    Filter JSONL rows by tokenized length of the "output" field.

    INPUT_JSONL_DIR: Directory containing *.jsonl files
    OUTPUT_DIR: Directory to write filtered JSONL files (same filenames)
    TOKENIZER_NAME_OR_PATH: Name or path for AutoTokenizer
    """
    jsonl_files: List[Path] = sorted(input_jsonl_dir.glob("*.jsonl"))
    if not jsonl_files:
        click.echo(f"[WARN] No *.jsonl files found in: {input_jsonl_dir}")
        return

    # Print tokenizer info once (for visibility)
    additional_special_tokens = _prepare_additional_tokens(add_special_tokens)
    tok = _load_tokenizer(tokenizer_name_or_path, additional_special_tokens)
    click.echo(f"tokenizer.special_tokens_map: {tok.special_tokens_map}")
    click.echo(
        f"Processing {len(jsonl_files)} files with max_token_len={max_token_len}, "
        f"num_worker={num_worker}"
    )

    totals = {
        "input_rows": 0,
        "output_rows": 0,
        "removed_rows": 0,
        "invalid_rows": 0,
    }

    output_dir = Path(output_dir)
    if output_dir.exists():
        click.echo(f"[WARN] Output dir already exists: {output_dir}, removing...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parallel: one process per file
    tasks: List[Tuple] = []
    with ProcessPoolExecutor(max_workers=num_worker) as ex:
        futures = []
        for p in jsonl_files:
            futures.append(
                ex.submit(
                    _process_single_file,
                    str(p),
                    str(output_dir),
                    tokenizer_name_or_path,
                    additional_special_tokens,
                    max_token_len,
                )
            )

        for fut in as_completed(futures):
            stats = fut.result()
            # Per-file report
            kept_pct = (
                (stats["output_rows"] / stats["input_rows"] * 100)
                if stats["input_rows"]
                else 0.0
            )
            removed_pct = (
                (stats["removed_rows"] / stats["input_rows"] * 100)
                if stats["input_rows"]
                else 0.0
            )
            invalid_pct = (
                (stats["invalid_rows"] / stats["input_rows"] * 100)
                if stats["input_rows"]
                else 0.0
            )

            click.echo(
                f"[{stats['file']}] "
                f"in={stats['input_rows']}  "
                f"out={stats['output_rows']} ({kept_pct:.1f}%)  "
                f"removed={stats['removed_rows']} ({removed_pct:.1f}%)  "
                f"invalid={stats['invalid_rows']} ({invalid_pct:.1f}%)  "
                f"→ {stats['output_path']}"
            )

            # Accumulate totals
            totals["input_rows"] += stats["input_rows"]
            totals["output_rows"] += stats["output_rows"]
            totals["removed_rows"] += stats["removed_rows"]
            totals["invalid_rows"] += stats["invalid_rows"]

    # Grand total
    if totals["input_rows"] > 0:
        kept_pct = totals["output_rows"] / totals["input_rows"] * 100
        removed_pct = totals["removed_rows"] / totals["input_rows"] * 100
        invalid_pct = totals["invalid_rows"] / totals["input_rows"] * 100
    else:
        kept_pct = removed_pct = invalid_pct = 0.0

    click.echo("\n=== TOTALS ===")
    click.echo(
        f"in={totals['input_rows']}  "
        f"out={totals['output_rows']} ({kept_pct:.1f}%)  "
        f"removed={totals['removed_rows']} ({removed_pct:.1f}%)  "
        f"invalid={totals['invalid_rows']} ({invalid_pct:.1f}%)"
    )


if __name__ == "__main__":
    main()
