"""
python scripts/tools/rm_ckpt_keep_last.py \
     ~/mnt/workspace/svg_glyph_llm/saves/250910-google_font-ablate_scale/Qwen3-14B-rel_coord \
    --keep_last \
    --dry_run

find ~/mnt/workspace/svg_glyph_llm/saves/250915-envato-upper_bound \
    -type d -name 'checkpoint-*' -prune -o \
    -type d -name 'runs' -print \
    -exec bash -c 'python scripts/tools/rm_ckpt_keep_last.py "$(dirname "$1")"' _ {} \;
"""

import logging
import os
import shutil
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("output_dir", type=str, required=True)
@click.option("--keep_last/--no_keep_last", is_flag=True, default=True)
@click.option("--dry_run/--no_dry_run", is_flag=True, default=False)
def main(output_dir, keep_last=True, dry_run=False):
    logging.info(f"output_dir: {output_dir}")
    logging.info(f"keep_last: {keep_last}")
    logging.info(f"dry_run: {dry_run}")

    output_dir = Path(output_dir)
    if not output_dir.exists():
        err_msg = f"Output directory {output_dir} does not exist."
        logger.error(err_msg)
        raise ValueError(err_msg)

    trainer_state_path = output_dir / "trainer_state.json"
    if trainer_state_path.exists():
        logger.info(f"Found trainer_state.json, assuming training is finished.")
    else:
        err_msg = f"trainer_state.json does not exist, assuming training is not finished. Skipping."
        logger.warning(err_msg)
        exit(0)

    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    logger.info(f"Found {len(checkpoints)} checkpoints.")
    if len(checkpoints) == 0:
        logger.info("No checkpoints found, skipping.")
        exit(0)

    # Keep the last one
    if keep_last:
        last = checkpoints[-1]
        checkpoints = checkpoints[:-1]
        logger.info(f"Keeping last ckpt: {last}...")

    logger.info(f"Deleting {len(checkpoints)} checkpoints...")
    if dry_run:
        logger.info("Dry run, skipping deletion.")
        return

    for ckpt in checkpoints:
        print(f"Deleting {ckpt}...")
        shutil.rmtree(ckpt)

    print("✅ Finish deleting ckpts.")


if __name__ == "__main__":
    main()
