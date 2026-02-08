"""
python -m src.tools.check_two_jsonl_key \
-a misc/250903-alphanumeric-ref_img/ood_font_family_decon \
-b misc/250903-alphanumeric-ref_img-eval_baselines/ood_font_family_decon \
-k images

python -m src.tools.check_two_jsonl_key \
-a misc/250903-alphanumeric-ref_img-b64_pil/ood_font_family_decon \
-b misc/250903-alphanumeric-ref_img-eval_baselines-b64_pil/ood_font_family_decon \
-k images
"""

import logging
from pathlib import Path

import click
import tqdm
from src.svg_glyph_gen_v2.utils import load_jsonl, load_jsonl_by_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--jsonl_path_or_dir_a", "-a", type=click.Path(exists=True), required=True
)
@click.option(
    "--jsonl_path_or_dir_b", "-b", type=click.Path(exists=True), required=True
)
@click.option("--check_key", "-k", type=str, default="images", help="Key to check")
@click.option("--by_generator", is_flag=True, default=False, help="Load by generator")
@click.option("--pdb", is_flag=True, default=False, help="Enable pdb")
def main(jsonl_path_or_dir_a, jsonl_path_or_dir_b, check_key, by_generator, pdb):
    print(f"Loading {jsonl_path_or_dir_a} and {jsonl_path_or_dir_b}")
    data_a = load_data(jsonl_path_or_dir_a, by_generator, logger=logger)
    data_b = load_data(jsonl_path_or_dir_b, by_generator, logger=logger)
    for sample_a, sample_b in tqdm.tqdm(zip(data_a, data_b)):
        sample_a_value = sample_a[check_key]
        sample_b_value = sample_b[check_key]
        if sample_a_value != sample_b_value:
            if pdb:
                from IPython import embed

                embed()
            raise ValueError(
                f"Found different {check_key} for {sample_a_value} and {sample_b_value}"
            )

    print(f"All key ({check_key}) are the same!")
    if pdb:
        # fmt: off
        from IPython import embed; embed()
        # fmt: on


def load_data(jsonl_path_or_dir, by_generator, logger):
    if by_generator:
        return load_jsonl_by_generator(jsonl_path_or_dir, logger=logger)
    return load_jsonl(jsonl_path_or_dir, logger=logger)


if __name__ == "__main__":
    main()
