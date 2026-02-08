"""
python misc/download_outputs_from_storage.py [INPUT_STORAGE_PATH] [--output_base_dir OUTPUT_BASE_DIR] [--storage_prefix_path STORAGE_PREFIX_PATH]

python misc/download_outputs_from_storage.py workspace/svg_glyph_llm/outputs/250919-eval-gfont-baselines
"""

import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.argument("input_storage_path", type=str, required=True)
@click.option("--output_base_dir", type=str, default="./")
@click.option(
    "--storage_prefix_path",
    type=str,
    default="workspace/svg_glyph_llm/",
)
@click.option("--jobs", type=int, default=10)
@click.option("--threads", type=int, default=20)
@click.option("--overwrite/--no_overwrite", default=True)
@click.option("--dry_run/--no_dry_run", default=False)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    logger.info(f"args: {args}")

    storage_prefix_path = Path(args.storage_prefix_path)
    output_base_dir = Path(args.output_base_dir)
    input_storage_path = Path(args.input_storage_path)

    output_relative_path = input_storage_path.relative_to(storage_prefix_path)
    output_path = output_base_dir / output_relative_path

    logger.info(f"output_path: {output_path}")

    command = []
    command.extend(
        "storage_cli --prod-use-cython-client getr --using_direct_reads".split()
    )

    command.extend(["--jobs", str(args.jobs)])
    command.extend(["--threads", str(args.threads)])

    command.extend([str(input_storage_path), str(output_path)])

    if not args.overwrite:
        command.append("--skip_already_downloaded")

    logger.info(f"command: {' '.join(command)}")
    if args.dry_run:
        logger.info("Dry run mode, skip downloading.")
        return

    subprocess.run(command, check=True)
    logger.info("Done.")


if __name__ == "__main__":
    main()
