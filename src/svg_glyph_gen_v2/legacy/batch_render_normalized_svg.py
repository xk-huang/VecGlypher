"""
python -m src.svg_glyph_gen_v2.batch_render_normalized_svg --num_workers=20 --max_num_fonts=10 --max_num_content=10  --pass_idx=1 --overwrite True
python -m src.svg_glyph_gen_v2.batch_render_normalized_svg --num_workers=20 --max_num_fonts=10 --max_num_content=10  --pass_idx=2 --apply_translate False
python -m src.svg_glyph_gen_v2.batch_render_normalized_svg --num_workers=20 --max_num_fonts=10 --max_num_content=10  --pass_idx=3

python -m src.svg_glyph_gen_v2.batch_render_normalized_svg --num_workers=20 --pass_idx=1 --overwrite True
python -m src.svg_glyph_gen_v2.batch_render_normalized_svg --num_workers=20 --pass_idx=2 --apply_translate False
python -m src.svg_glyph_gen_v2.batch_render_normalized_svg --num_workers=20 --pass_idx=3

Inspect occrupted svg:
input_content_file=data/processed/content/alphanumeric.txt
python -m src.svg_glyph_gen_v2.inspect_svg_hash --input_content_file $input_content_file
"""

import json
import shutil
from concurrent.futures import as_completed, ProcessPoolExecutor
from functools import lru_cache
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import click
import pandas as pd
import tqdm
from fontTools.ttLib import TTFont

from .filter_by_pangram_svg import blake2_hash
from .render_normalized_svg import apply_transform_to_svg, round_optim_svg, text_to_svg
from .utils import load_jsonl, setup_logger


@click.command()
@click.option(
    "--input_metadata_jsonl",
    type=click.Path(exists=True),
    default="data/processed/google_font_metadata/google_font_metadata.filter_invalid.filter_by_pangram_svg.jsonl",
)
@click.option(
    "--input_content_file",
    type=click.Path(exists=True),
    default="data/processed/content/alphanumeric.txt",
)
@click.option(
    "--input_google_font_dir",
    type=click.Path(exists=True),
    default="data/google_fonts/ofl",
)
@click.option("--content_split_tsv", default=None, help="Path to contents split file.")
@click.option("--font_split_tsv", default=None, help="Path to fonts split file.")
@click.option(
    "--output_dir", type=click.Path(), default="data/processed/normalized_svg"
)
@click.option("--batch_size", type=int, default=1000)
@click.option("--num_workers", type=int, default=20)
@click.option("--overwrite", type=bool, default=False)
@click.option("--apply_scale", type=bool, default=True)
@click.option("--apply_translate", type=bool, default=False)
@click.option("--pass_idx", type=int, default=1)
@click.option("--max_num_fonts", type=int, default=None)
@click.option("--max_num_content", type=int, default=None)
@click.option("--font_size", type=int, default=5000)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    batch_renderer = BatchRenderer(args)
    batch_renderer.run()


class BatchRenderer:
    def __init__(self, args):
        self.args = args

        self.font_sub_dir = Path(self.args.input_google_font_dir)

        output_dir = Path(self.args.output_dir)
        self.output_dir = output_dir
        if output_dir.exists() and args.overwrite:
            print(f"Overwrite {output_dir}. Removing it.")
            shutil.rmtree(output_dir)

        for sub_dir in ["raw", "xfm", "round_optim"]:
            (output_dir / sub_dir).mkdir(parents=True, exist_ok=True)

        self.output_raw_dir = self.output_dir / "raw"
        self.output_xfm_dir = self.output_dir / "xfm"
        self.output_round_optim_dir = self.output_dir / "round_optim"

        self.logger = setup_logger(self.output_dir)

        self.apply_scale = args.apply_scale
        self.apply_translate = args.apply_translate
        self.num_workers = args.num_workers
        self.pass_idx = args.pass_idx
        self.font_size = args.font_size

        if self.pass_idx not in [1, 2, 3]:
            raise ValueError(f"Invalid pass_idx: {self.pass_idx}")
        if self.pass_idx != 1 and args.overwrite:
            raise ValueError(
                "Overwrite is only supported for pass_idx=1. "
                "Please set --overwrite=False for pass_idx=2 or 3."
            )

        metadata_list = self.load_metadata()
        content_list = self.load_content()
        if args.max_num_fonts is not None:
            metadata_list = metadata_list[: args.max_num_fonts]
            self.logger.info(f"Truncated metadata_list to {len(metadata_list)}")
        if args.max_num_content is not None:
            content_list = content_list[: args.max_num_content]
            self.logger.info(f"Truncated content_list to {len(content_list)}")
        self.metadata_list = metadata_list
        self.content_list = content_list

        self.logger.info(f"apply_scale: {self.apply_scale}")
        self.logger.info(f"apply_translate: {self.apply_translate}")
        self.logger.info(f"num_workers: {self.num_workers}")
        self.logger.info(f"pass_idx: {self.pass_idx}")
        self.logger.info(f"font_size: {self.font_size}")

    def run(self):
        self.check_hash_font_path_content()

        self.render_font_content(
            self.metadata_list[0], self.content_list[0], verbose=True
        )

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for metadata in self.metadata_list:
                futures.append(
                    executor.submit(
                        self.render_font_content, metadata, self.content_list
                    )
                )
            for result in tqdm.tqdm(as_completed(futures), total=len(futures)):
                if result.exception():
                    self.logger.error(result.exception())
                    continue
        self.logger.info("Done.")

        if self.pass_idx == 1:
            return
        elif self.pass_idx == 2:
            self.compare_dir_a_b(self.output_raw_dir, self.output_xfm_dir)
        elif self.pass_idx == 3:
            self.compare_dir_a_b(self.output_xfm_dir, self.output_round_optim_dir)
        else:
            raise ValueError(f"Invalid pass_idx: {self.pass_idx}")

    def compare_dir_a_b(self, dir_a, dir_b):
        dir_a_files_set = set([i.name for i in dir_a.glob("*.svg")])
        dir_b_files_set = set([i.name for i in dir_b.glob("*.svg")])

        dir_a_only_files = dir_a_files_set - dir_b_files_set
        dir_b_only_files = dir_b_files_set - dir_a_files_set
        if dir_a_only_files or dir_b_only_files:
            err_msg = f"Output dir mismatch. {dir_a} vs {dir_b}"
            self.logger.error(err_msg)
            self.logger.error(f"dir_a_only_files: {dir_a_only_files}")
            self.logger.error(f"dir_b_only_files: {dir_b_only_files}")
            raise ValueError(err_msg)
        self.logger.info(f"Output dir match between {dir_a} and {dir_b}")

    @lru_cache(maxsize=512)
    def load_font(self, font_family_dir_name, font_file_name):
        font_path = self.font_sub_dir / font_family_dir_name / font_file_name
        return TTFont(font_path)

    def render_font_content(self, metadata, content_list, verbose=False):
        font_family_dir_name = metadata["font_family_dir_name"]
        font_file_name = metadata["filename"]
        if verbose:
            self.logger.info(f"Rendering {font_family_dir_name}/{font_file_name}")

        font = self.load_font(font_family_dir_name, font_file_name)

        for content in content_list:
            output_hash = hash_font_content(
                font_family_dir_name, font_file_name, content
            )
            if verbose:
                self.logger.info(f"Rendering {content} to {output_hash}")
            try:
                self.render_normalized_svg_for_font_content(font, content, output_hash)
            except Exception as e:
                self.logger.error(f"Failed to render {content} to {output_hash}: {e}")

    def render_normalized_svg_for_font_content(
        self, font_path, content, output_file_name
    ):
        output_file_name = f"{output_file_name}.svg"

        output_raw_path = self.output_raw_dir / output_file_name
        output_xfm_path = self.output_xfm_dir / output_file_name
        output_round_optim_path = self.output_round_optim_dir / output_file_name

        if self.pass_idx == 1:
            text_to_svg(font_path, content, output_raw_path, font_size=self.font_size)
        elif self.pass_idx == 2:
            apply_transform_to_svg(
                output_raw_path,
                output_xfm_path,
                apply_scale=self.apply_scale,
                apply_translate=self.apply_translate,
            )
        elif self.pass_idx == 3:
            round_optim_svg(output_xfm_path, output_round_optim_path)
        else:
            raise ValueError(f"Invalid pass_idx: {self.pass_idx}")

    def check_hash_font_path_content(self):
        existing_hashes = set()
        is_confict = False
        for metadata, content in product(self.metadata_list, self.content_list):
            font_file_name = metadata["filename"]
            font_family_dir_name = metadata["font_family_dir_name"]
            output_hash = hash_font_content(
                font_family_dir_name, font_file_name, content
            )

            if output_hash in existing_hashes:
                self.logger.warning(
                    f"Hash {output_hash} already exists. Skipping {font_family_dir_name}/{font_file_name} and {content}"
                )
                is_confict = True
                continue
            existing_hashes.add(output_hash)
        if is_confict:
            self.logger.warning("Hash conflict detected.")
            raise ValueError("Hash conflict detected.")
        else:
            self.logger.info("No hash conflict detected.")

    def load_metadata(self):
        metadata_list = []
        input_metadata_jsonl = self.args.input_metadata_jsonl
        input_metadata_jsonl = Path(input_metadata_jsonl)
        if not input_metadata_jsonl.exists():
            raise FileNotFoundError(f"{input_metadata_jsonl} does not exist.")

        metadata_list = load_jsonl(input_metadata_jsonl, logger=self.logger)

        split_tsv = self.args.font_split_tsv
        if split_tsv is not None:
            metadata_list = self._use_split_tsv(split_tsv, metadata_list)

        self.logger.info(
            f"Loaded {len(metadata_list)} metadata from {input_metadata_jsonl}"
        )

        return metadata_list

    def load_content(self):
        content_list = []
        input_content_file = self.args.input_content_file
        input_content_file = Path(input_content_file)

        if not input_content_file.exists():
            raise FileNotFoundError(f"{input_content_file} does not exist.")
        with open(input_content_file, "r") as f:
            for line in f:
                content_list.append(line.strip())

        split_tsv = self.args.content_split_tsv
        if split_tsv is not None:
            content_list = self._use_split_tsv(split_tsv, content_list)

        self.logger.info(
            f"Loaded {len(content_list)} content from {input_content_file}"
        )

        return content_list

    def _use_split_tsv(self, split_tsv, input_list):
        self.logger.info(f"Reading split from {split_tsv}")
        split_df = pd.read_csv(split_tsv, sep="\t")
        index = split_df["index"].tolist()
        self.logger.info(f"Number of items: {len(input_list)} -> {len(index)}")
        input_list = [input_list[i] for i in index]
        return input_list


def hash_font_content(font_family_dir_name, font_file_name, content):
    input_str = f"{font_family_dir_name}|{font_file_name}|{content}"
    return blake2_hash(input_str)


if __name__ == "__main__":
    main()
