"""
Copied from: src/svg_glyph_gen_v2/batch_render_normalized_svg.py
Save svg in chunked jsonl


python -m src.svg_glyph_gen_v2.batch_render_normalized_svg_v2 --num_workers=20 \
    --max_num_fonts=10 --max_num_content=10 --overwrite True

python -m src.svg_glyph_gen_v2.batch_render_normalized_svg_v2 --num_workers=20  --overwrite True


Inspect occrupted svg:
input_content_file=data/processed/content/alphanumeric.txt
python -m src.svg_glyph_gen_v2.inspect_svg_hash --input_content_file $input_content_file
"""

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
from IPython import embed

from .filter_by_pangram_svg import blake2_hash
from .render_normalized_svg import (
    apply_transform_to_svg_str,
    round_rm_space_svg_path,
    text_to_svg,
)
from .utils import load_jsonl, prepare_output_dir_and_logger, write_jsonl


@click.command()
@click.option(
    "--input_metadata_jsonl",
    type=click.Path(exists=True),
    # default="data/processed/google_font_metadata-filter_invalid-filter_by_pangram",
    required=True,
)
@click.option(
    "--input_content_file",
    type=click.Path(exists=True),
    # default="data/processed/content/alphanumeric.txt",
    required=True,
)
@click.option(
    "--input_google_font_dir",
    type=click.Path(exists=True),
    # default="data/google_fonts/ofl",
    required=True,
)
@click.option("--content_split_tsv", default=None, help="Path to contents split file.")
@click.option("--font_split_tsv", default=None, help="Path to fonts split file.")
@click.option(
    "--output_dir",
    type=click.Path(),
    # default="data/processed/normalized_svg",
    required=True,
)
@click.option("--batch_size", type=int, default=1000)
@click.option("--num_workers", type=int, default=20)
@click.option("--apply_scale", type=bool, default=True)
@click.option("--apply_translate", type=bool, default=True)
@click.option("--max_num_fonts", type=int, default=None)
@click.option("--max_num_content", type=int, default=None)
# NOTE: 1000 for typical em-square size
@click.option("--font_size", type=int, default=1000)
# NOTE: round to 1 decimal place, there are 0.5 pixel difference
@click.option("--decimals", type=int, default=1)
@click.option("--use_relative_path", type=bool, default=True)
# NOTE: make ascender and descender consistent across all fonts
@click.option("--final_ascender", type=int, default=1000)
@click.option("--final_descender", type=int, default=-300)
# Other options
@click.option("--metadata_batch_size", type=int, default=40)
@click.option("--content_batch_size", type=int, default=100)
@click.option("--debug", is_flag=True, default=False)
@click.option("--overwrite", is_flag=True, default=False)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    batch_renderer = BatchRenderer(args)
    batch_renderer.run()


class BatchRenderer:
    def __init__(self, args):
        self.args = args

        # prepare output dir and logger
        should_skip, logger = prepare_output_dir_and_logger(
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        if should_skip:
            exit()
        self.output_dir = Path(args.output_dir)
        self.logger = logger

        self.font_sub_dir = Path(self.args.input_google_font_dir)
        self.output_path = self.output_dir / "normalized_svg.jsonl"

        self.apply_scale = args.apply_scale
        self.apply_translate = args.apply_translate
        self.num_workers = args.num_workers
        self.font_size = args.font_size
        self.decimals = args.decimals
        self.use_relative_path = args.use_relative_path
        self.final_ascender = args.final_ascender
        self.final_descender = args.final_descender

        self.logger.info(f"apply_scale: {self.apply_scale}")
        self.logger.info(f"apply_translate: {self.apply_translate}")
        self.logger.info(f"num_workers: {self.num_workers}")
        self.logger.info(f"font_size: {self.font_size}")
        self.logger.info(f"decimals: {self.decimals}")
        self.logger.info(f"use_relative_path: {self.use_relative_path}")
        self.logger.info(f"final_ascender: {self.final_ascender}")
        self.logger.info(f"final_descender: {self.final_descender}")

    def run(self):
        metadata_list = self.load_metadata()
        content_list = self.load_content()

        if self.args.max_num_fonts is not None:
            metadata_list = metadata_list[: self.args.max_num_fonts]
            self.logger.info(f"Truncated metadata_list to {len(metadata_list)}")
        if self.args.max_num_content is not None:
            content_list = content_list[: self.args.max_num_content]
            self.logger.info(f"Truncated content_list to {len(content_list)}")

        metadata_batch_size = self.args.metadata_batch_size
        content_batch_size = self.args.content_batch_size
        self.logger.info(f"metadata_batch_size: {metadata_batch_size}")
        self.logger.info(f"content_batch_size: {content_batch_size}")

        self.check_hash_font_path_content(
            metadata_list=metadata_list, content_list=content_list
        )

        # test run for debugging
        _ = self.render_font_content(metadata_list[:1], content_list[:3], verbose=True)
        if self.args.debug:
            self.logger.info("Debug mode. Exiting.")
            embed()
            return

        output_svg_dict_list = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for metadata_start_idx in range(0, len(metadata_list), metadata_batch_size):
                # NOTE: metadata_batch_size is the key for efficiency.
                # if the amount metadata is too large and the amount content is too small,
                # we can increase the number of metadata per process to imporve throughput.
                metadata_end_idx = min(
                    metadata_start_idx + metadata_batch_size,
                    len(metadata_list),
                )
                metadata_batch = metadata_list[metadata_start_idx:metadata_end_idx]

                for content_start_idx in range(
                    0, len(content_list), content_batch_size
                ):
                    content_end_idx = min(
                        content_start_idx + content_batch_size,
                        len(content_list),
                    )
                    content_batch = content_list[content_start_idx:content_end_idx]
                    futures.append(
                        executor.submit(
                            self.render_font_content, metadata_batch, content_batch
                        )
                    )

            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="iterate futures"
            ):
                if future.exception():
                    self.logger.error(future.exception())
                    continue
                output_svg_dict_list.extend(future.result())

        self.logger.info("Saving to jsonl...")
        write_jsonl(output_svg_dict_list, self.output_path, logger=self.logger)
        self.logger.info("Done.")

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
        return TTFont(font_path, lazy=True)

    def render_font_content(self, metadata_batch, content_list, verbose=False):
        output_svg_dict_list = []

        for metadata in metadata_batch:
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
                    output_svg = self.render_normalized_svg_for_font_content(
                        font, content, output_hash
                    )
                    output_svg_dict_list.append(
                        {"hash": output_hash, "svg": output_svg}
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to render {content} to {output_hash}: {e}.\nWe skip this font/content pair without raising exception"
                    )
        return output_svg_dict_list

    def render_normalized_svg_for_font_content(
        self, font_path, content, output_file_name
    ):
        output_file_name = f"{output_file_name}.svg"

        svg = text_to_svg(
            font_path,
            content,
            None,
            font_size=self.font_size,
            final_ascender=self.final_ascender,
            final_descender=self.final_descender,
        )
        xfm_svg = apply_transform_to_svg_str(
            svg,
            apply_scale=self.apply_scale,
            apply_translate=self.apply_translate,
            use_relative_path=self.use_relative_path,
        )
        if xfm_svg is None:
            raise ValueError(f"Failed to apply transform to {content}")

        round_optim_svg = round_rm_space_svg_path(xfm_svg, self.decimals)
        if self.args.debug:
            print(f"svg: {svg}")
            print(f"xfm_svg: {xfm_svg}")
            print(f"round_optim_svg: {round_optim_svg}")
        return round_optim_svg

    def check_hash_font_path_content(self, metadata_list, content_list):
        existing_hashes = set()
        is_confict = False
        for metadata, content in product(metadata_list, content_list):
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
