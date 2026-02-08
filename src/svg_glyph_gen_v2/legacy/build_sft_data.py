"""
llama-factory dataset format: https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md#alpaca-format

python -m src.svg_glyph_gen_v2.build_sft_data \
    --content_split_tsv data/processed/split_train_test_index/alphanumeric/contents/train.tsv \
    --font_split_tsv data/processed/split_train_test_index/alphanumeric/fonts/train.tsv \
    --output_dir "data/processed/sft/alphanumeric-train_f-train_c"
"""

import json
import random

import shutil
from concurrent.futures import as_completed, ProcessPoolExecutor
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import click
import pandas as pd
import tqdm

from .batch_render_normalized_svg import hash_font_content
from .svg_simplifier import SVGSimplifier
from .utils import setup_logger

SYSTEM_PROMPT = """You are a glyph designer that outputs **only** SVG <path> elements.

Hard constraints:
- Output one <path> per glyph, in reading order of the given text.
- Each <path> must use only two attributes: d and transform.
- The transform attribute, if present, must be exactly: transform="translate(x,y)".
- Use translate for positioning.
- Your output must be raw <path> lines only, each terminated by a newline.
"""

STYLE_TEMPLATE = """Font design requirements:
{style_str}
"""

CONTENT_TEMPLATE = """Text content:
{content_str}
"""


WORD_SEP = "<|SEP|>"


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
    "--input_svg_dir",
    type=click.Path(exists=True),
    default="data/processed/normalized_svg/round_optim",
)
@click.option(
    "--output_dir", default="data/processed/sft/alphanumeric-all_fonts", type=str
)
@click.option("--output_log_dir", default="data/processed/sft", type=str)
@click.option("--content_split_tsv", default=None, help="Path to contents split file.")
@click.option("--font_split_tsv", default=None, help="Path to fonts split file.")
# attributes
@click.option("--num_workers", default=40, type=int)
@click.option("--chunk_size", default=10000, type=int)
@click.option("--max_dataset_size", default=None, type=int)
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite existing output."
)
@click.option("--apply_metadata_tags", type=bool, default=True)
@click.option("--apply_group_tags", type=bool, default=True)
@click.option("--apply_word_sep", type=bool, default=False)
@click.option("--verbose", is_flag=True, default=False)
@click.option("--seed", default=42, type=int)
@click.option("--num_fonts", default=None, type=int)
@click.option("--num_contents", default=None, type=int)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    random.seed(args.seed)
    sft_data_builder = SFTDataBuilder(args)
    sft_data_builder.run()


class InstrctionBuilder:
    def __init__(self):
        pass

    def __call__(
        self,
        *,
        metadata,
        svg_str,
        content_str,
        apply_metadata_tags=True,
        apply_group_tags=True,
        apply_word_sep=True,
        verbose=False,
        other_metadata=None,
    ):
        category = metadata.get("category", None)
        classifications = metadata.get("classifications", None)
        stroke = metadata.get("stroke", None)
        style = metadata.get("style", None)
        weight = metadata.get("weight", None)

        # metadata tags
        if apply_metadata_tags:
            _other_tags = []
            if category is not None:
                if isinstance(category, list):
                    _other_tags.extend([f"{self._format_text(i)}" for i in category])
                else:
                    raise ValueError(f"Unknown type for category: {type(category)}")
            if classifications is not None:
                if isinstance(classifications, list):
                    _other_tags.extend(
                        [f"{self._format_text(i)}" for i in classifications]
                    )
                else:
                    raise ValueError(
                        f"Unknown type for classifications: {type(classifications)}"
                    )
            if stroke is not None:
                if isinstance(stroke, str):
                    _other_tags.append(self._format_text(stroke))
                else:
                    raise ValueError(f"Unknown type for stroke: {type(stroke)}")

            # deduplicate other tags
            _other_tags = list(set(_other_tags))

            metadata_tags = [
                f"{style} style",
                f"{weight} weight",
                *_other_tags,
            ]
        else:
            metadata_tags = []

        # human accessed expressive and typographic tags
        if apply_group_tags:
            tags = metadata["tags"]
        else:
            tags = []

        # unique and shuffle all tags
        all_tags = list(set(metadata_tags + tags))
        all_tags = random.sample(all_tags, len(all_tags))

        # build instruction
        style_str = ", ".join(all_tags)

        if apply_word_sep:
            content_str = self._sep_word(content_str)

        formatted_content_str = CONTENT_TEMPLATE.format(content_str=content_str)
        if style_str:
            style_str = STYLE_TEMPLATE.format(style_str=style_str)
            instruction_str = "\n\n".join([style_str, formatted_content_str])
        else:
            instruction_str = formatted_content_str

        if verbose:
            print(f"\n========\n{instruction_str}\n========\n")

        sft_row = {
            "instruction": instruction_str,
            "system": SYSTEM_PROMPT,
            "output": svg_str,
        }

        # NOTE: save sft inputs to metadata
        metadata.update({f"sft_{k}": v for k, v in sft_row.items()})
        metadata["content_str"] = content_str
        if other_metadata is not None:
            metadata.update(other_metadata)

        # NOTE: save dumped metadata to sft row
        sft_row.update({"metadata": json.dumps(metadata)})
        return sft_row

    @staticmethod
    def _format_text(text):
        """
        SANS_SERIF -> sans-serif
        """
        text = text.replace("_", "-")
        text = text.lower()
        return text

    @staticmethod
    def _sep_word(content_str):
        """
        "Yes" -> "Y<|think|>e<|think|>s"
        """
        content_str = WORD_SEP.join(list(content_str))
        return content_str


class SFTDataBuilder:
    def __init__(self, args):
        self.args = args

        output_dir = Path(self.args.output_dir)
        self.output_dir = output_dir
        if args.overwrite:
            print(f"Overwrite {output_dir}. Removing it.")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(args.output_log_dir)

        self.metadata_list = self.load_metadata()
        self.content_list = self.load_content()
        total_num_rows = len(self.metadata_list) * len(self.content_list)
        self.logger.info(f"Total number of rows: {total_num_rows}")

        self.input_svg_dir = Path(self.args.input_svg_dir)
        self.svg_simplifier = SVGSimplifier()
        self.instruction_builder = InstrctionBuilder()

    def run(self):
        # test one run
        verbose = False
        content = self.content_list[0]
        metadata = self.metadata_list[0]
        svg_str = self.build_svg_for_font_content(metadata, content, verbose=verbose)

        sft_dict = self.instruction_builder(
            metadata=metadata,
            svg_str=svg_str,
            content_str=content,
            verbose=verbose,
        )

        if verbose:
            import pprint

            pprint.pprint(sft_dict)

        # batch inference
        # NOTE(xk): single process is faster than multi-process
        self._parallel_process()

    def _single_process(self):
        self.chunk_size = self.args.chunk_size
        self.max_dataset_size = self.args.max_dataset_size
        self.current_size = 0
        self.chunk_idx = 0

        self.apply_metadata_tags = self.args.apply_metadata_tags
        self.apply_group_tags = self.args.apply_group_tags

        self.sft_dict_list = []
        self.num_saved_rows = 0
        for metadata, content in tqdm.tqdm(
            product(self.metadata_list, self.content_list),
            total=len(self.metadata_list) * len(self.content_list),
            desc="Processing metadata and content",
        ):
            if (
                self.max_dataset_size is not None
                and self.current_size >= self.max_dataset_size
            ):
                break

            sft_dict = self._build_sft_dict(metadata, content)
            self.sft_dict_list.append(sft_dict)
            self.current_size += 1

            if len(self.sft_dict_list) >= self.args.chunk_size:
                self._save_sft_jsonl()
        if len(self.sft_dict_list) >= self.args.chunk_size:
            self._save_sft_jsonl()
        self.logger.info(f"Saved {self.current_size} SFT data to {self.output_dir}")

    def _parallel_process(self):
        self.chunk_size = self.args.chunk_size
        self.max_dataset_size = self.args.max_dataset_size
        self.current_size = 0
        self.chunk_idx = 0

        self.apply_metadata_tags = self.args.apply_metadata_tags
        self.apply_group_tags = self.args.apply_group_tags
        self.logger.info(f"Apply metadata tags: {self.apply_metadata_tags}")
        self.logger.info(f"Apply group tags: {self.apply_group_tags}")

        futures = []
        num_workers = self.args.num_workers
        self.logger.info(f"Using {num_workers} workers.")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for metadata_content_arg_list in self._metadata_content_arg_iter():
                futures.append(
                    executor.submit(
                        self._parallel_process_worker, metadata_content_arg_list
                    )
                )

        self.sft_dict_list = []
        self.num_saved_rows = 0
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            if future.exception():
                self.logger.error(future.exception())
                continue
            sft_dict_list = future.result()
            self.sft_dict_list.extend(sft_dict_list)
            self._save_sft_jsonl()
            self.logger.info(f"Saved {self.chunk_idx} chunks.")

    def _parallel_process_worker(self, metadata_content_arg_list):
        sft_dict_list = []
        for metadata, content in tqdm.tqdm(
            metadata_content_arg_list, desc="worker processing"
        ):
            sft_dict = self._build_sft_dict(metadata, content)
            sft_dict_list.append(sft_dict)
        return sft_dict_list

    def _metadata_content_arg_iter(self):
        chunk_size = self.args.chunk_size
        args_list = []
        for idx, (metadata, content) in enumerate(
            product(self.metadata_list, self.content_list)
        ):
            if self.max_dataset_size is not None and idx >= self.max_dataset_size:
                if len(args_list) > 0:
                    yield args_list
                    break
            if idx > 0 and idx % chunk_size == 0:
                yield args_list
                args_list = []
            args_list.append((metadata, content))
        if len(args_list) > 0:
            yield args_list

    def _build_sft_dict(self, metadata, content):
        svg_str, svg_path = self.build_svg_for_font_content(
            metadata, content, return_path=True
        )
        sft_dict = self.instruction_builder(
            metadata=metadata,
            svg_str=svg_str,
            content_str=content,
            apply_metadata_tags=self.apply_metadata_tags,
            apply_group_tags=self.apply_group_tags,
            other_metadata={"svg_path": str(svg_path)},
        )
        return sft_dict

    def _save_sft_jsonl(self):
        output_sft_jsonl_path = self.output_dir / f"{self.chunk_idx:05d}.jsonl"
        with open(output_sft_jsonl_path, "w", buffering=8192 * 16) as f:
            for sft_dict in self.sft_dict_list:
                f.write(json.dumps(sft_dict) + "\n")
        self.chunk_idx += 1
        self.num_saved_rows += len(self.sft_dict_list)
        self.logger.info(
            f"Saved chunk {self.chunk_idx} (total {self.num_saved_rows} rows) to: {output_sft_jsonl_path}"
        )

        self.sft_dict_list.clear()

    def build_svg_for_font_content(
        self, metadata, content, verbose=False, return_path=False
    ):
        font_family_dir_name = metadata["font_family_dir_name"]
        font_file_name = metadata["filename"]
        if verbose:
            self.logger.info(f"Rendering {font_family_dir_name}/{font_file_name}")

        svg_hash = hash_font_content(font_family_dir_name, font_file_name, content)
        if verbose:
            self.logger.info(f"Rendering {content} to {svg_hash}")

        svg_path = self.input_svg_dir / f"{svg_hash}.svg"
        if not svg_path.exists():
            raise FileNotFoundError(f"{svg_path} does not exist.")

        svg_str = svg_path.read_text()
        tokenized_svg_str = self.svg_simplifier.encode(svg_str)
        detokenized_svg_str = self.svg_simplifier.decode(tokenized_svg_str)
        if self.svg_simplifier.encode(detokenized_svg_str) != tokenized_svg_str:
            raise ValueError(
                f"Detokenized SVG string does not match the original SVG string. {detokenized_svg_str} != {svg_str}"
            )
        if verbose:
            self.logger.info(f"SVG string: {svg_str}")
            self.logger.info(f"Tokenized SVG string: {tokenized_svg_str}")
            self.logger.info(f"Detokenized SVG string: {detokenized_svg_str}")

        if return_path:
            return tokenized_svg_str, svg_path

        return tokenized_svg_str

    def load_metadata(self):
        metadata_list = []
        input_metadata_jsonl = self.args.input_metadata_jsonl
        input_metadata_jsonl = Path(input_metadata_jsonl)
        if not input_metadata_jsonl.exists():
            raise FileNotFoundError(f"{input_metadata_jsonl} does not exist.")

        with open(input_metadata_jsonl, "r") as f:
            for line in f:
                metadata_list.append(json.loads(line))

        split_tsv = self.args.font_split_tsv
        if split_tsv is not None:
            metadata_list = self._use_split_tsv(split_tsv, metadata_list)

        self.logger.info(
            f"Loaded {len(metadata_list)} metadata from {input_metadata_jsonl}"
        )

        # sample fonts if needed
        num_fonts = self.args.num_fonts
        if num_fonts is not None:
            prev_len = len(metadata_list)
            if num_fonts > len(metadata_list):
                err_msg = f"num_fonts {num_fonts} is larger than the number of fonts {len(metadata_list)}"
                self.logger.error(err_msg)
                raise ValueError(err_msg)

            metadata_list = random.sample(metadata_list, num_fonts)
            self.logger.info(
                f"Randomly sample {prev_len} -> {len(metadata_list)} fonts."
            )

        return metadata_list

    def _use_split_tsv(self, split_tsv, input_list):
        self.logger.info(f"Reading split from {split_tsv}")
        split_df = pd.read_csv(split_tsv, sep="\t")
        index = split_df["index"].tolist()
        self.logger.info(f"Number of items: {len(input_list)} -> {len(index)}")
        input_list = [input_list[i] for i in index]
        return input_list

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

        # sample contents if needed
        num_contents = self.args.num_contents
        if num_contents is not None:
            prev_len = len(content_list)
            if num_contents > len(content_list):
                err_msg = f"num_contents {num_contents} is larger than the number of contents {len(content_list)}"
                self.logger.error(err_msg)
                raise ValueError(err_msg)

            content_list = random.sample(content_list, num_contents)
            self.logger.info(
                f"Randomly sample {prev_len} -> {len(content_list)} contents."
            )

        return content_list


if __name__ == "__main__":
    main()
