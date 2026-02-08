"""
python -m src.svg_glyph_gen_v2.extract_metadata_tags
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import click
import pandas as pd
import tqdm
from google.protobuf import json_format, text_format

from .fonts_public_pb2 import FamilyProto
from .utils import prepare_output_dir_and_logger


@click.command()
@click.option(
    "--input_google_font_dir",
    type=Path,
    # default="data/google_fonts/ofl",
    required=True,
)
@click.option(
    "--input_google_font_metadata_dir",
    type=Path,
    # default="data/google_fonts/tags/",
    required=True,
)
@click.option(
    "--output_dir",
    type=Path,
    # default="data/processed/google_font_metadata",
    required=True,
)
@click.option("--overwrite", is_flag=True, default=False)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    extractor = GoogleFontMetadataTagsExtractor(args)
    extractor.test()

    extractor.run()


class GoogleFontMetadataTagsExtractor:
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

        # load tags for metadata
        self.input_google_font_metadata_dir = Path(args.input_google_font_metadata_dir)
        self.tag_df = self._load_tag()

        # load font file base dir
        font_sub_dir = Path(args.input_google_font_dir)
        font_family_dir_name_list = [p.name for p in font_sub_dir.iterdir()]
        self.font_sub_dir = font_sub_dir
        self.font_family_dir_name_list = font_family_dir_name_list

    def run(self):
        output_metadata_path = self.output_dir / "google_font_metadata.jsonl"

        font_metadata_jsonl = []
        failed_font_family_dir_name_list = []

        for font_family_dir_name in tqdm.tqdm(self.font_family_dir_name_list):
            try:
                _font_metadata_jsonl = self.extract_dump_metadata(font_family_dir_name)
            except Exception as e:
                self.logger.warning(f"Failed to process {font_family_dir_name}: {e}")
                failed_font_family_dir_name_list.append(font_family_dir_name)
                continue
            font_metadata_jsonl.extend(_font_metadata_jsonl)

        num_success_fonts = len(font_metadata_jsonl)
        num_failed_fonts = len(failed_font_family_dir_name_list)
        self.logger.info(f"num_success_fonts: {num_success_fonts}")
        self.logger.info(f"num_failed_fonts: {num_failed_fonts}")

        self.logger.info(f"Writing to: {output_metadata_path}")
        with open(output_metadata_path, "w") as f:
            for line in font_metadata_jsonl:
                f.write(line + "\n")
        self.logger.info(f"Done writing to: {output_metadata_path}")

    def test(self):
        # test read
        font_family_dir_name_list = self.font_family_dir_name_list
        font_family_dir_name = font_family_dir_name_list[1]

        font_metadata_jsonl = self.extract_dump_metadata(font_family_dir_name)
        self.logger.info(f"font_metadata_jsonl: {font_metadata_jsonl}")

    def extract_dump_metadata(self, font_family_dir_name):
        font_family_metadata = self._load_font_family_metadata_tag(font_family_dir_name)
        font_metadata_list = self._load_fonts_from_family(font_family_metadata)
        font_metadata_jsonl = self._convert_to_jsonl(font_metadata_list)
        return font_metadata_jsonl

    def _convert_to_jsonl(self, font_metadata_list):
        return [json.dumps(i) for i in font_metadata_list]

    def _load_fonts_from_family(self, font_family_metadata):
        name = font_family_metadata["name"]

        num_fonts = len(font_family_metadata["fonts"])
        font_family_metadata["num_fonts"] = num_fonts

        font_metadata_list = []
        for font_data in font_family_metadata["fonts"]:
            font_data_name = font_data["name"]

            if font_data_name != name:
                raise ValueError(
                    f"Font name mismatch: {name} != {font_data_name} in {font_family_metadata}"
                )

            font_metadata = font_family_metadata.copy()
            font_metadata.pop("fonts")

            for k, v in font_data.items():
                if k == "name":
                    continue
                if k in font_family_metadata:
                    raise ValueError(f"Duplicate key {k} in font data and metadata")
                font_metadata[k] = v
            font_metadata_list.append(font_metadata)
        return font_metadata_list

    def _load_tags_for_font_family(self, metadata):
        font_family_name = metadata["name"]

        tags = self.tag_df.loc[font_family_name]["tags"]
        tag_weights = self.tag_df.loc[font_family_name]["tag_weight"]

        if isinstance(tags, pd.Series):
            tags = tags.to_list()
        else:
            tags = [tags]

        if isinstance(tag_weights, pd.Series):
            tag_weights = tag_weights.to_list()
        else:
            tag_weights = [tag_weights]

        metadata["tags"] = tags
        metadata["tag_weights"] = tag_weights

        return metadata

    def _load_font_family_metadata_tag(self, font_family_dir_name):
        metadata = self._load_google_font_metadata(font_family_dir_name)
        metadata = self._load_tags_for_font_family(metadata)

        return metadata

    def _load_google_font_metadata(self, font_family_dir_name):
        font_sub_dir = self.font_sub_dir
        font_family_metadata_path = font_sub_dir / font_family_dir_name / "METADATA.pb"
        if not font_family_metadata_path.exists():
            raise ValueError(
                f"Font metadata file does not exist: {font_family_metadata_path}"
            )

        family_proto = FamilyProto()

        with open(font_family_metadata_path, "r", encoding="utf-8") as fh:
            text_format.Parse(fh.read(), family_proto)

        metadata = json_format.MessageToDict(family_proto)
        metadata["font_family_dir_name"] = font_family_dir_name
        metadata["source"] = "google_font"
        return metadata

    TAG_VERSIONS = ["old", "new"]

    def _load_tag(self):
        tag_dir = self.input_google_font_metadata_dir
        tag_csv_dir = tag_dir / "all"
        tag_version = "new"

        if tag_version == "old":
            file_name = "families.csv"
            tags_path = tag_csv_dir / file_name
            # tags_path = local2storage(tags_path)
            with open(str(tags_path), "r") as f:
                df = pd.read_csv(f, index_col=0, header=None)
        elif tag_version == "new":
            family_name = "families_new.csv"
            tags_path = tag_csv_dir / family_name
            # tags_path = local2storage(tags_path)
            with open(tags_path, "r") as f:
                df = pd.read_csv(f, index_col=0, header=None)

            # remove the first column, since they are empty
            # https://github.com/google/fonts/blob/main/tags/all/families.csv
            df = df.iloc[:, 1:]
        else:
            raise ValueError(f"version must be one of {self.TAG_VERSIONS}")

        df.columns = ["raw_tags", "tag_weight"]
        df.index.name = "font_family"

        # map raw_tags to tags
        tags_metadata_csv = tag_dir / "tags_metadata.csv"
        tags_metadata_df = pd.read_csv(tags_metadata_csv, header=None)
        tags_metadata_df.columns = ["raw_tags", "low", "high", "tags"]

        raw_tags2tags = dict(
            zip(tags_metadata_df["raw_tags"], tags_metadata_df["tags"])
        )
        # manually fix "/Sans/Grotesque" and "/Sans/Superellipse"
        raw_tags2tags["Sans/Grotesque"] = "grotesque sans-serif"
        raw_tags2tags["Sans/Superellipse"] = "superellipse sans-serif"
        df["tags"] = df["raw_tags"].map(raw_tags2tags)

        return df


if __name__ == "__main__":
    main()
