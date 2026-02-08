"""
python -m src.tools.load_jsonl
"""

import logging
from pathlib import Path

import click
from src.svg_glyph_gen_v2.utils import load_jsonl, load_jsonl_by_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("jsonl_path_or_dir", type=click.Path(exists=True), required=True)
@click.option("--by_generator", is_flag=True, default=False, help="Load by generator")
@click.option("--pdb", is_flag=True, default=False, help="Enable pdb")
def main(jsonl_path_or_dir, by_generator, pdb):
    jsonl_path_or_dir = Path(jsonl_path_or_dir)
    if by_generator:
        for _ in load_jsonl_by_generator(jsonl_path_or_dir, logger=logger):
            pass
    else:
        _ = load_jsonl(jsonl_path_or_dir, logger=logger)

    print("Done!")
    if pdb:
        # fmt: off
        from IPython import embed; embed()
        # fmt: on


if __name__ == "__main__":
    main()
