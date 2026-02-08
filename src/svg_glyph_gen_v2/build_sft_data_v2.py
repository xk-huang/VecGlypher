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
from typing import Optional

import click
import pandas as pd
import tqdm
from IPython import embed

from .batch_render_normalized_svg_v2 import hash_font_content
from .svg_simplifier import SVGSimplifier
from .utils import load_jsonl, prepare_output_dir_and_logger

SYSTEM_PROMPT = """You are a specialized vector glyph designer creating SVG path elements.

CRITICAL REQUIREMENTS:
- Each glyph must be a complete, self-contained <path> element, in reading order of the given text.
- Terminate each <path> element with a newline character
- Output ONLY valid SVG <path> elements
"""

STYLE_TEMPLATE = """Font design requirements: {style_str}
"""

CONTENT_TEMPLATE = """Text content: {content_str}
"""

WORD_SEP = "<|SEP|>"

SVG_REPR_TYPES = ["simplified", "original"]


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
    "--input_svg_dir",
    type=click.Path(exists=True),
    # default="data/processed/normalized_svg/",
    required=True,
)
@click.option(
    "--output_dir",
    type=str,
    # default="data/processed/sft/alphanumeric-all_fonts",
    required=True,
)
@click.option(
    "--output_log_dir",
    type=str,
    # default="data/processed/sft",
    required=True,
)
@click.option("--content_split_tsv", default=None, help="Path to contents split file.")
@click.option("--font_split_tsv", default=None, help="Path to fonts split file.")
# attributes
@click.option("--num_workers", default=20, type=int)
@click.option("--chunk_size", default=10000, type=int)
@click.option("--max_dataset_size", default=None, type=int)
@click.option("--apply_metadata_tags", type=bool, default=True)
@click.option("--apply_group_tags", type=bool, default=True)
@click.option("--apply_word_sep", type=bool, default=True)
@click.option("--seed", default=42, type=int)
@click.option("--num_fonts", default=None, type=int)
@click.option("--num_contents", default=None, type=int)
@click.option("--enable_worker_tqdm", is_flag=True, default=False)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    random.seed(args.seed)
    sft_data_builder = SFTDataBuilder(args)
    sft_data_builder.run()


class InstructionBuilder:
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
            # NOTE: there is an newline at the end of style and content template
            instruction_str = "\n".join([style_str, formatted_content_str])
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
        # NOTE: avoid update in for loop, otherwise it will raise error
        # make a shallow copy of metadata
        metadata = metadata.copy()
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
        "Yes" -> "Y<|SEP|>e<|SEP|>s"
        """
        content_str = WORD_SEP.join(list(content_str))
        return content_str


G_HASH2SVG: Optional[dict] = None
G_SVG_SIMPLIFIER: Optional[SVGSimplifier] = None
G_INSTR_BUILDER: Optional[InstructionBuilder] = None


def _init_worker(input_svg_dir: str):
    """
    Initialize per-process globals. On Linux (fork), these may already be
    inherited; we only build them if missing. On spawn (mac/Win), we load here.
    """
    from pathlib import Path

    global G_HASH2SVG, G_SVG_SIMPLIFIER, G_INSTR_BUILDER

    if G_HASH2SVG is None:
        svg_list = load_jsonl(Path(input_svg_dir))
        G_HASH2SVG = {svg["hash"]: svg["svg"] for svg in svg_list}

    if G_SVG_SIMPLIFIER is None:
        G_SVG_SIMPLIFIER = SVGSimplifier()

    if G_INSTR_BUILDER is None:
        G_INSTR_BUILDER = InstructionBuilder()


def build_svg_for_font_content(
    metadata, content, verbose=False, logger=None, return_path=False
):
    font_family_dir_name = metadata["font_family_dir_name"]
    font_file_name = metadata["filename"]
    if verbose and logger:
        logger.info(f"Rendering {font_family_dir_name}/{font_file_name}")

    svg_hash = hash_font_content(font_family_dir_name, font_file_name, content)
    if verbose and logger:
        logger.info(f"Rendering {content} to {svg_hash}")

    if G_HASH2SVG.get(svg_hash, None) is None:
        raise ValueError(f"SVG {svg_hash} does not exist.")

    svg_str = G_HASH2SVG.get(svg_hash)
    tokenized_svg_str = G_SVG_SIMPLIFIER.encode(svg_str)
    detokenized_svg_str = G_SVG_SIMPLIFIER.decode(tokenized_svg_str)
    if G_SVG_SIMPLIFIER.encode(detokenized_svg_str) != tokenized_svg_str:
        raise ValueError(
            f"Detokenized SVG string does not match the original SVG string. {detokenized_svg_str} != {svg_str}"
        )
    if verbose and logger:
        logger.info(f"SVG string: {svg_str}")
        logger.info(f"Tokenized SVG string: {tokenized_svg_str}")
        logger.info(f"Detokenized SVG string: {detokenized_svg_str}")

    if return_path:
        return tokenized_svg_str, svg_hash

    return tokenized_svg_str


def _build_sft_dict(*, metadata, content, apply_metadata_tags, apply_group_tags):
    svg_str, svg_path = build_svg_for_font_content(metadata, content, return_path=True)
    sft_dict = G_INSTR_BUILDER(
        metadata=metadata,
        svg_str=svg_str,
        content_str=content,
        apply_metadata_tags=apply_metadata_tags,
        apply_group_tags=apply_group_tags,
        other_metadata={"svg_path": str(svg_path)},
    )
    return sft_dict


def _parallel_process_worker(
    metadata_content_arg_list,
    apply_metadata_tags,
    apply_group_tags,
    enable_worker_tqdm,
    logger=None,
):
    sft_dict_list = []
    for metadata, content in tqdm.tqdm(
        metadata_content_arg_list,
        desc="worker batch processing",
        disable=not enable_worker_tqdm,
    ):
        # NOTE: avoid update in for loop, otherwise it will raise error
        # make a shallow copy of metdata
        metadata = metadata.copy()
        try:
            sft_dict = _build_sft_dict(
                metadata=metadata,
                content=content,
                apply_metadata_tags=apply_metadata_tags,
                apply_group_tags=apply_group_tags,
            )
        except Exception as e:
            if logger:
                logger.error(
                    f"Error build sft dict: {e}.\nWe skip this font/content pair without raising exception"
                )
            continue

        sft_dict_list.append(sft_dict)
    return sft_dict_list


class SFTDataBuilder:
    def __init__(self, args):
        self.args = args

        # prepare output dir and logger
        should_skip, logger = prepare_output_dir_and_logger(
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            output_log_dir=args.output_log_dir,
        )
        if should_skip:
            exit()
        self.output_dir = Path(args.output_dir)
        self.logger = logger

        self.input_svg_dir = Path(self.args.input_svg_dir)

    def build_hash2svg(self):
        svg_list = load_jsonl(self.input_svg_dir, logger=self.logger)
        hash2svg = {}
        for svg in svg_list:
            hash2svg[svg["hash"]] = svg["svg"]
        self.logger.info(f"Loaded {len(hash2svg)} SVGs from {self.input_svg_dir}")
        return hash2svg

    def run(self):
        # Make the heavy read-only state visible to forked children immediately.
        # (On spawn platforms, _init_worker will construct them.)
        _init_worker(self.input_svg_dir)

        verbose = self.args.verbose
        debug = self.args.debug

        # test one run
        metadata_list = self.load_metadata()
        content_list = self.load_content()
        total_num_rows = len(metadata_list) * len(content_list)
        self.logger.info(f"Total number of rows: {total_num_rows}")

        # test run for debugging
        content = content_list[0]
        metadata = metadata_list[0]
        svg_str = build_svg_for_font_content(metadata, content, verbose=verbose)

        sft_dict = G_INSTR_BUILDER(
            metadata=metadata,
            svg_str=svg_str,
            content_str=content,
            verbose=verbose,
        )

        if verbose:
            import pprint

            pprint.pprint(sft_dict)
        if debug:
            embed()

        # batch inference
        self._parallel_process(metadata_list, content_list)

    def _single_process(self, metadata_list, content_list):
        self.chunk_size = self.args.chunk_size
        self.max_dataset_size = self.args.max_dataset_size
        self.current_size = 0
        self.chunk_idx = 0

        self.apply_metadata_tags = self.args.apply_metadata_tags
        self.apply_group_tags = self.args.apply_group_tags

        sft_dict_list = []
        self.num_saved_rows = 0
        for metadata, content in tqdm.tqdm(
            product(metadata_list, content_list),
            total=len(metadata_list) * len(content_list),
            desc="Processing metadata and content",
        ):
            if (
                self.max_dataset_size is not None
                and self.current_size >= self.max_dataset_size
            ):
                break

            sft_dict = _build_sft_dict(
                metadata=metadata,
                content=content,
                apply_metadata_tags=self.apply_metadata_tags,
                apply_group_tags=self.apply_group_tags,
            )
            sft_dict_list.append(sft_dict)
            self.current_size += 1

            if len(sft_dict_list) >= self.args.chunk_size:
                self._save_sft_jsonl(sft_dict_list)
                sft_dict_list = []

        # save if there is remaining data
        if len(sft_dict_list) > 0:
            self._save_sft_jsonl(sft_dict_list)
            sft_dict_list = []
        self.logger.info(f"Saved {self.current_size} SFT data to {self.output_dir}")

    def _parallel_process(self, metadata_list, content_list):
        self.chunk_size = self.args.chunk_size
        self.max_dataset_size = self.args.max_dataset_size
        self.current_size = 0
        self.chunk_idx = 0

        self.apply_metadata_tags = self.args.apply_metadata_tags
        self.apply_group_tags = self.args.apply_group_tags
        self.logger.info(f"Apply metadata tags: {self.apply_metadata_tags}")
        self.logger.info(f"Apply group tags: {self.apply_group_tags}")

        num_workers = self.args.num_workers
        self.logger.info(f"Using {num_workers} workers.")

        self.logger.info("Submitting jobs...")
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(str(self.input_svg_dir),),
        ) as executor:
            futures = []
            for metadata_content_arg_list in self._metadata_content_arg_iter(
                metadata_list, content_list
            ):
                futures.append(
                    executor.submit(
                        _parallel_process_worker,
                        metadata_content_arg_list,
                        self.apply_metadata_tags,
                        self.apply_group_tags,
                        self.args.enable_worker_tqdm,
                        self.logger,
                    )
                )
            self.logger.info(
                f"Submitted {len(futures)} jobs. Waiting for completion..."
            )

            self.num_saved_rows = 0
            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="jobs processing"
            ):
                if future.exception():
                    self.logger.error(future.exception())
                    raise future.exception()
                sft_dict_list = future.result()
                if sft_dict_list:  # Only save if there is data
                    self._save_sft_jsonl(sft_dict_list)
                    self.logger.info(f"Saved {self.chunk_idx} chunks.")

    def _metadata_content_arg_iter(self, metadata_list, content_list):
        chunk_size = self.args.chunk_size
        args_list = []
        num_samples = len(metadata_list) * len(content_list)
        if self.max_dataset_size is not None:
            self.logger.info(f"Limiting dataset size to {self.max_dataset_size}")
            num_samples = min(num_samples, self.max_dataset_size)
        self.logger.info(f"Number of samples: {num_samples}")
        for idx, (metadata, content) in enumerate(
            tqdm.tqdm(
                product(metadata_list, content_list),
                desc="args building",
                total=num_samples,
            )
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

    def _save_sft_jsonl(self, sft_dict_list):
        output_sft_jsonl_path = self.output_dir / f"{self.chunk_idx:05d}.jsonl"
        with open(output_sft_jsonl_path, "w", buffering=8192 * 16) as f:
            for sft_dict in sft_dict_list:
                f.write(json.dumps(sft_dict) + "\n")
        self.chunk_idx += 1
        self.num_saved_rows += len(sft_dict_list)
        self.logger.info(
            f"Saved chunk {self.chunk_idx} (total {self.num_saved_rows} rows) to: {output_sft_jsonl_path}"
        )

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
