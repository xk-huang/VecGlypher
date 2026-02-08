"""
python scripts/tools/download_model_from_storage.py \
    --exp_name 250910-google_font-ablate_svg_repr \
    --model_path Qwen3-1_7B-rel_coord/checkpoint-21680

python scripts/tools/download_model_from_storage.py [-i] [-o]

NOTE: we only download the file in the input ckpt dir, and skip all subdirs.
"""

import logging
import re
import shutil
import subprocess
import sys
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


@click.command(
    help="Download model from storage_cli to local. "
    "If --input_storage_model_path and --output_model_path are specified, directly download model. "
    "Otherwise, download model from storage_ckpt_saves_dir/exp_name/model_path to output_dir/exp_name/model_path."
)
@click.option(
    "--storage_ckpt_saves_dir",
    type=str,
    default="workspace/svg_glyph_llm/saves",
)
@click.option("--exp_name", type=str, default=None)
@click.option("--model_path", type=str, default=None)
@click.option("--output_dir", type=str, default="saves/")
# directly download model
@click.option("--input_storage_model_path", "-i", type=str, default=None)
@click.option("--output_model_path", "-o", type=str, default=None)
# other flags
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite existing files"
)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    overwrite = args.overwrite

    input_storage_model_path = args.input_storage_model_path
    output_model_path = args.output_model_path
    if Path(output_model_path).exists() and not overwrite:
        logger.info(
            f"Output model path {output_model_path} already exists. Use --overwrite to overwrite."
        )
        return

    if input_storage_model_path is None and output_model_path is None:
        output_dir = Path(args.output_dir)
        storage_ckpt_saves_dir = Path(args.storage_ckpt_saves_dir)
        exp_name = args.exp_name
        model_path = args.model_path
        if exp_name is None or model_path is None:
            raise ValueError(
                "Please specify both exp_name and model_path. Or use -i and -o."
            )

        # make sure model_path is valid
        model_path = model_path.lstrip("/").rstrip("/")
        input_storage_model_path = storage_ckpt_saves_dir / exp_name / model_path

        output_model_path = output_dir / exp_name / model_path
    elif input_storage_model_path is not None and output_model_path is not None:
        input_storage_model_path = Path(input_storage_model_path)
        output_model_path = Path(output_model_path)
    else:
        raise ValueError(
            "Please specify both input_storage_model_path and output_model_path. Or specify neither."
        )
    output_model_path.mkdir(parents=True, exist_ok=True)

    # get ckpt files
    logger.info(f"download model from storage_cli args: {args}")
    logger.info(f"Downloading {input_storage_model_path} to {output_model_path}")
    command = ["storage_cli", "--prod-use-cython-client", "ls", input_storage_model_path]
    result = subprocess.run(command, text=True, capture_output=True, check=True)

    # NOTE: only download files
    last_fields = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        line_split = line.split()
        if len(line_split) != 2:
            raise ValueError(f"Invalid line: {line}, not 2 fields")
        first_part, last_part = line_split
        if first_part.isdigit():
            last_fields.append(last_part)

    # NOTE: use regex to filter out some files
    regex_pattern = re.compile(
        r"^(rng_state.*|scheduler\.pt|global_step.*|checkpoint-[0-9]+)$"
    )
    ckpt_files = [x for x in last_fields if not regex_pattern.match(x)]

    # download files
    retries = 1
    threads = 20
    jobs = 10
    quiet = True
    workers = 8

    def task(name: str):
        src = input_storage_model_path / name
        dst = output_model_path / name
        attempts = retries + 1
        for _ in range(attempts):
            status, return_str = storage_get(
                src, dst, threads=threads, jobs=jobs, quiet=quiet, overwrite=overwrite
            )
            if status != "fail":
                return status, return_str
        return "fail", name

    total = len(ckpt_files)
    done_cnt = skip_cnt = fail_cnt = 0
    logger.info(f"Downloading {total} files with {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(task, n): n for n in ckpt_files}
        processed = 0
        for fut in as_completed(futs):
            status, name = fut.result()
            processed += 1
            if status == "done":
                done_cnt += 1
                msg = f"done {name}"
            elif status == "skip":
                skip_cnt += 1
                msg = f"skip {name} (exists)"
            else:
                fail_cnt += 1
                msg = f"fail {name}"
            # lightweight progress line
            logger.info(f"[{processed}/{total}] {msg}")

    logger.info(f"\nSummary: done={done_cnt}, skip={skip_cnt}, fail={fail_cnt}")
    if fail_cnt:
        logger.error(f"Failed to download {fail_cnt} files")
        sys.exit(2)


def storage_get(
    src: str, dst: Path, threads: int, jobs: int, quiet: bool, overwrite: bool
) -> tuple[str, str]:
    """
    Download one file with storage_cli get.
    Returns (status, name) where status in {"done", "skip", "fail"}.
    """

    if dst.exists() and not overwrite:
        return ("skip", dst.name)

    cmd = [
        "storage_cli",
        "--prod-use-cython-client",
        "get",
        "--using_direct_reads",
        "--threads",
        str(threads),
        "-j",
        str(jobs),
        src,
        str(dst),
    ]
    # quiet mode like `2>/dev/null` unless it fails
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None

    try:
        subprocess.run(cmd, check=True, stdout=stdout, stderr=stderr)
        size = dst.stat().st_size
        return_str = dst.name + f" ({sizeof_fmt(size)})"
        return ("done", return_str)
    except subprocess.CalledProcessError:
        return ("fail", dst.name)


def sizeof_fmt(num: int, suffix="B") -> str:
    num = float(num)
    for unit in ["", "K", "M", "G", "T"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"


if __name__ == "__main__":
    main()
