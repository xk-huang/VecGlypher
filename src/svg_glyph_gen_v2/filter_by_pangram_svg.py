"""
python -m src.svg_glyph_gen_v2.filter_by_pangram_svg
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import load_jsonl, prepare_output_dir_and_logger, write_jsonl


@click.command()
@click.option(
    "--input_metadata_jsonl",
    type=click.Path(exists=True),
    # default="data/processed/google_font_metadata-filter_invalid",
    required=True,
)
@click.option(
    "--input_pangram_jsonl",
    type=click.Path(exists=True),
    # default="data/processed/google_font_metadata-filter_invalid-pangram",
    required=True,
)
@click.option(
    "--output_dir",
    type=click.Path(),
    # default="data/processed/google_font_metadata-filter_invalid-filter_by_pangram",
    required=True,
)
@click.option("--quantile", type=float, default=0.90)
@click.option("--num_workers", type=int, default=20)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def main(**kargs):
    args = SimpleNamespace(**kargs)

    font_filter = PangramSVGFilter(args)
    font_filter.run()


class PangramSVGFilter:
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

        self.num_workers = args.num_workers

    def run(self):
        metadata = self.load_metadata()
        pangram_svgs = self.load_pangram_svgs()

        # make sure the pangram_svgs and metadata are in the same order
        identifier_list = [i["identifier"] for i in metadata]
        if len(set(identifier_list)) != len(identifier_list):
            raise ValueError("Duplicate identifier in metadata, cannot align order")
        identifier_list = [i["identifier"] for i in pangram_svgs]
        if len(set(identifier_list)) != len(identifier_list):
            raise ValueError("Duplicate identifier in pangram_svgs, cannot align order")

        metadata = sorted(metadata, key=lambda x: x["identifier"])
        pangram_svgs = sorted(pangram_svgs, key=lambda x: x["identifier"])
        for i, j in zip(metadata, pangram_svgs):
            if i["identifier"] != j["identifier"]:
                raise ValueError("Inconsistent identifier")

        metadata, pangram_svgs = self.filter_long_and_zero(metadata, pangram_svgs)
        for i, j in zip(metadata, pangram_svgs):
            if i["identifier"] != j["identifier"]:
                raise ValueError("Inconsistent identifier")
        metadata, pangram_svgs = self.filter_duplicate_paths(metadata, pangram_svgs)
        for i, j in zip(metadata, pangram_svgs):
            if i["identifier"] != j["identifier"]:
                raise ValueError("Inconsistent identifier")

        input_metadata_jsonl_name = Path(self.args.input_metadata_jsonl)
        output_filtered_metadata_jsonl = (
            self.output_dir
            / input_metadata_jsonl_name.with_suffix(".filter_by_pangram_svg.jsonl").name
        )
        write_jsonl(metadata, output_filtered_metadata_jsonl, logger=self.logger)
        self.logger.info(f"Write to: {output_filtered_metadata_jsonl}")

    def filter_duplicate_paths(self, metadata, pangram_svgs):
        from svgpathtools import svgstr2paths

        deduplicator = Deduplicator(logger=self.logger)
        is_unique_list = []
        duplicated_font_file_names = []
        for pangram_svg in tqdm(pangram_svgs, desc="Deduplicate"):
            svg = pangram_svg["svg"]
            identifier = pangram_svg["identifier"]

            _, attributes = svgstr2paths(svg)
            d_str = ""
            for attribute in attributes:
                d_str += attribute["d"]

            is_unique, seen_identifier = deduplicator.is_unique(
                d_str, metadata=identifier
            )
            is_unique_list.append(is_unique)
            if not is_unique:
                duplicated_font_file_names.append(identifier)
                # self.logger.warning(
                #     f"Duplicate path: (Input) {font_file_name}\t(Seen) {seen_font_file_name}"
                # )
            if self.args.verbose:
                self.logger.info(
                    f"Font file name: {identifier}. seen_font_file_name: {seen_identifier}. Unique: {is_unique}."
                )

        output_dup_dir = self.output_dir / "duplicated"
        output_dup_dir.mkdir(parents=True, exist_ok=True)
        output_dup_path = output_dup_dir / "duplicated_font_file_names.txt"
        duplicated_font_file_names = sorted(duplicated_font_file_names)
        with open(output_dup_path, "w") as f:
            for font_file_name in duplicated_font_file_names:
                f.write(font_file_name + "\n")
        self.logger.info(f"Write to: {output_dup_path}")

        # NOTE: final guardrail. Make sure 100% no mismatch of identifier
        num_mismatch = 0
        for i, j in zip(metadata, pangram_svgs):
            if i["identifier"] != j["identifier"]:
                num_mismatch += 1
        self.logger.warning(f"Number of identifier mismatch: {num_mismatch}")
        if num_mismatch > 0:
            raise ValueError("Number of identifier mismatch > 0")

        filtered_metadata = []
        filtered_pangram_svgs = []
        for metadata_item, pangram_svg, is_unique in zip(
            metadata, pangram_svgs, is_unique_list
        ):
            if pangram_svg["identifier"] != metadata_item["identifier"]:
                raise ValueError("Inconsistent identifier")

            if not is_unique:
                continue
            filtered_metadata.append(metadata_item)
            filtered_pangram_svgs.append(pangram_svg)

        len_before = len(metadata)
        len_after = len(filtered_metadata)
        self.logger.info(
            f"[Deduplication] Number of fonts: {len_before} -> {len_after}"
        )

        return filtered_metadata, filtered_pangram_svgs

    def filter_long_and_zero(self, metadata, pangram_svgs):
        identifier2svg_len = {i["identifier"]: i["svg_len"] for i in pangram_svgs}
        quantile = self.args.quantile
        svg_len_threshold = np.quantile(list(identifier2svg_len.values()), quantile)

        svg_len_series = pd.Series(identifier2svg_len.values())
        pd.options.display.float_format = "{:,.0f}".format
        self.logger.info(f"SVG len stats: {svg_len_series.describe()}")
        self.logger.info(f"SVG len threshold: {svg_len_threshold} @ {quantile}")

        filtered_metadata = [
            i
            for i in metadata
            if (identifier2svg_len[i["identifier"]] < svg_len_threshold)
            and (identifier2svg_len[i["identifier"]] > 0)
        ]
        filtered_pangram_svgs = [
            i
            for i in pangram_svgs
            if i["svg_len"] < svg_len_threshold and i["svg_len"] > 0
        ]

        len_before = len(metadata)
        len_after = len(filtered_metadata)
        self.logger.info(
            f"[Filter long and zero SVG] Number of fonts: {len_before} -> {len_after}"
        )
        return filtered_metadata, filtered_pangram_svgs

    def load_metadata(self):
        input_metadata_jsonl = self.args.input_metadata_jsonl
        data = load_jsonl(input_metadata_jsonl)
        for item in data:
            item["identifier"] = item["font_family_dir_name"] + "/" + item["filename"]
        return data

    def load_pangram_svgs(self):
        input_pangram_jsonl = self.args.input_pangram_jsonl
        data = load_jsonl(input_pangram_jsonl)
        for item in data:
            item["identifier"] = item["font_family_dir_name"] + "/" + item["filename"]
        return data


def blake2_hash(data: str, *, digest_len=16) -> bytes:
    return hashlib.blake2b(data.encode("utf-8"), digest_size=digest_len).hexdigest()[
        :16
    ]


class Deduplicator:
    def __init__(self, hasher=blake2_hash, logger=None):
        self.seen_hash2str = {}
        self.seen_hash2metadata = {}
        self.hasher = hasher

        self.logger = logger
        if logger:
            logger.info("Deduplicator initialized")

    def clear(self):
        self.seen_hash2str = {}
        self.seen_hash2metadata = {}
        if self.logger:
            self.logger.info("Deduplicator cleared")

    def is_unique(self, input_str, metadata=None):
        """
        Return True if input_str is unique, otherwise return False
        as well as the metadata of the existing input_str
        """
        output_hash = self.hasher(input_str)
        if output_hash in self.seen_hash2str:
            if input_str == self.seen_hash2str[output_hash]:
                return False, self.seen_hash2metadata[output_hash]
            else:
                if self.logger:
                    self.logger.warning(
                        f"Hash collision: (Input) {input_str}\t(Seen) {self.seen_hash2str[output_hash]}"
                    )
        self.seen_hash2str[output_hash] = input_str
        self.seen_hash2metadata[output_hash] = metadata
        return True, metadata


if __name__ == "__main__":
    main()
