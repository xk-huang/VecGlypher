"""
Input svg hash, get the font and content.

input_content_file=data/processed/content/alphanumeric.txt
python -m src.svg_glyph_gen_v2.inspect_svg_hash --input_content_file $input_content_file
"""

import json
import shutil
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import click
import tqdm

from .batch_render_normalized_svg import hash_font_content

from .utils import setup_logger


@click.command()
@click.option(
    "--input_metadata_jsonl",
    type=click.Path(exists=True),
    # default="data/processed/google_font_metadata/google_font_metadata.filter_invalid.filter_by_pangram_svg.jsonl",
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
@click.option(
    "--output_dir",
    type=click.Path(),
    # default="data/processed/inspect_svg_hash",
    required=True,
)
@click.option("--overwrite", type=bool, default=False)
@click.option("--apply_scale", type=bool, default=True)
@click.option("--apply_translate", type=bool, default=False)
@click.option("--pass_idx", type=int, default=1)
@click.option("--max_num_fonts", type=int, default=None)
@click.option("--max_num_content", type=int, default=None)
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

        self.logger = setup_logger(self.output_dir)

        self.apply_scale = args.apply_scale
        self.apply_translate = args.apply_translate
        self.pass_idx = args.pass_idx

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

    def run(self):
        self.check_hash_font_path_content()

        self.render_font_content(
            self.metadata_list[0], self.content_list[0], verbose=True
        )

        results = {}
        for metadata in tqdm.tqdm(self.metadata_list):
            results.update(self.render_font_content(metadata, self.content_list))

        # read from stdin
        print("Enter a line of text (or 'exit' / CTRL+C to exit):", end="")
        while True:
            print("Enter a line of text (or 'exit' / CTRL+C to exit):", end="")
            for line in sys.stdin:
                if line == "exit":
                    break
                line = line.strip()
                try:
                    print(f"{line}\t{results[line]}")
                except KeyError:
                    print(f"{line}\tNone")

    def render_font_content(self, metadata, content_list, verbose=False):
        font_family_dir_name = metadata["font_family_dir_name"]
        font_file_name = metadata["filename"]
        if verbose:
            self.logger.info(f"Rendering {font_family_dir_name}/{font_file_name}")

        hash2inputs = {}
        for content in content_list:
            output_hash = hash_font_content(
                font_family_dir_name, font_file_name, content
            )
            hash2inputs[output_hash] = (font_family_dir_name, font_file_name, content)
        return hash2inputs

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

        with open(input_metadata_jsonl, "r") as f:
            for line in f:
                metadata_list.append(json.loads(line))
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
        self.logger.info(
            f"Loaded {len(content_list)} content from {input_content_file}"
        )

        return content_list


if __name__ == "__main__":
    main()
