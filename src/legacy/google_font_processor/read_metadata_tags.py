"""
python -m  google_font_processor.read_metadata_tags
"""

import pprint
from pathlib import Path

from types import SimpleNamespace

import click

import pandas as pd

try:
    from genads.svg_vtg.google_font_processor import fonts_public_pb2
except ModuleNotFoundError:
    from . import fonts_public_pb2
from google.protobuf import json_format, text_format


def load_tags(tags_dir, version="old"):
    VERSIONS = ["old", "new"]
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}")

    if version == "old":
        family_name = "families.csv"
        tags_path = Path(tags_dir) / family_name
        # tags_path = local2storage(tags_path)
        with open(str(tags_path), "r") as f:
            df = pd.read_csv(f, index_col=0, header=None)

    elif version == "new":
        family_name = "families_new.csv"
        tags_path = Path(tags_dir) / family_name
        # tags_path = local2storage(tags_path)
        with open(tags_path, "r") as f:
            df = pd.read_csv(f, index_col=0, header=None)

        # remove the first column, since they are empty
        # https://github.com/google/fonts/blob/main/tags/all/families.csv
        df = df.iloc[:, 1:]
    else:
        raise ValueError(f"version must be one of {VERSIONS}")

    print(f"Loading {tags_path}")
    # add column names
    df.columns = ["group_tag", "group_tag_weight"]
    # A score between 0 and 100 (may have decimals) expressing how strongly the team
    # believes the tag applies to that family—100 = perfect exemplar,
    # lower numbers = looser fit.
    # Used to rank fonts inside each filter and to decide which tags surface for a family.
    # It is not the CSS font-weight; it’s a relevance/confidence score.
    df.index.name = "font_family"
    return df


def load_font_metadata(font_metadata_path, font_dir_name=None, verbose=False):
    family = fonts_public_pb2.FamilyProto()

    if verbose:
        print(f"Loading {font_metadata_path}")
    with open(font_metadata_path, "r", encoding="utf-8") as fh:
        text_format.Parse(fh.read(), family)

    metadata = json_format.MessageToDict(family)
    metadata["font_dir_name"] = font_dir_name

    metadata["source"] = "google_fonts"
    return metadata


def load_group_tags(metadata, group_tags_df):
    font_family_name = metadata["name"]

    group_tags = group_tags_df.loc[font_family_name]["group_tag"]
    group_tag_weights = group_tags_df.loc[font_family_name]["group_tag_weight"]
    if isinstance(group_tags, pd.Series):
        group_tags = group_tags.to_list()
    else:
        group_tags = [group_tags]
    if isinstance(group_tag_weights, pd.Series):
        group_tag_weights = group_tag_weights.to_list()
    else:
        group_tag_weights = [group_tag_weights]

    metadata["group_tags"] = group_tags
    metadata["group_tag_weights"] = group_tag_weights

    return metadata


@click.command()
@click.option("--tags_dir", default="data/google_fonts/tags/all/")
@click.option("--version", default="old")
@click.option("--font_dir", default="data/google_fonts/ofl")
@click.option("--font_dir_name", default="roboto")
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    tags_dir = args.tags_dir
    version = args.version
    df = load_tags(tags_dir, version=version)

    # load font metadata
    font_dir = args.font_dir
    font_dir_name = args.font_dir_name
    font_metadata_path = Path(font_dir) / font_dir_name / "METADATA.pb"
    metadata = load_font_metadata(font_metadata_path, verbose=True)

    font_name = metadata["name"]
    group_tags = df.loc[font_name]["group_tag"].to_list()
    group_tag_weights = df.loc[font_name]["group_tag_weight"].to_list()

    metadata["group_tags"] = group_tags
    metadata["group_tag_weights"] = group_tag_weights

    pprint.pprint(f"Font: {font_name} -> {metadata['name']}")
    pprint.pprint(metadata)


if __name__ == "__main__":
    main()
