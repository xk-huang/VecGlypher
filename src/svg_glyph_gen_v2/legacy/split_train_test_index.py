"""
Split train and val and outputs the index tsv file

python src/svg_glyph_gen_v2/split_train_test_index.py
"""

import json
from collections import defaultdict
from pathlib import Path

from pprint import pprint
from types import SimpleNamespace

import click
import numpy as np
import pandas as pd


@click.command()
@click.option(
    "--gfont_metadata_path",
    default="data/processed/google_font_metadata/google_font_metadata.filter_invalid.filter_by_pangram_svg.jsonl",
)
@click.option("--content_path", default="data/processed/content/alphanumeric.txt")
@click.option("--num_train_fonts", default=None, type=int)
@click.option("--num_ind_test_fonts", default=250, type=int)
@click.option("--num_ood_test_fonts", default=250, type=int)
@click.option("--num_train_contents", default=None, type=int)
@click.option("--num_ind_test_contents", default=10, type=int)
@click.option("--num_ood_test_contents", default=0, type=int)
@click.option(
    "--output_dir", default="data/processed/split_train_test_index/alphanumeric"
)
@click.option("--seed", default=42, type=int)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    gfont_metadata_path = getattr(args, "gfont_metadata_path")
    content_path = getattr(args, "content_path")

    # set random seed
    seed = getattr(args, "seed", 42)
    np.random.seed(seed)

    # load gfont metadata
    gfont_metadata_list = []
    with open(gfont_metadata_path, "r") as f:
        for line in f.readlines():
            gfont_metadata_list.append(json.loads(line))

    # load contents
    contents = []
    with open(content_path, "r") as f:
        for line in f.readlines():
            contents.append(line.strip())

    # split train and val
    num_fonts = len(gfont_metadata_list)
    num_contents = len(contents)

    num_ind_test_fonts = getattr(args, "num_ind_test_fonts", 200)
    num_ood_test_fonts = getattr(args, "num_ood_test_fonts", 200)

    num_ind_test_contents = getattr(args, "num_ind_test_contents", 200)
    num_ood_test_contents = getattr(args, "num_ood_test_contents", 200)

    num_train_fonts = getattr(args, "num_train_fonts", None)
    num_train_contents = getattr(args, "num_train_contents", None)
    if num_train_fonts is None:
        num_train_fonts = num_fonts - num_ood_test_fonts
    else:
        num_left = num_fonts - num_ood_test_fonts
        if num_train_fonts > num_left:
            raise ValueError(
                f"num_train_fonts > num_fonts: {num_train_fonts} > {num_left}"
            )
    if num_train_contents is None:
        num_train_contents = num_contents - num_ood_test_contents
    else:
        num_left = num_contents - num_ood_test_contents
        if num_train_contents > num_left:
            raise ValueError(
                f"num_train_contents > num_contents: {num_train_contents} > {num_left}"
            )

    split_arg_dict = {
        "fonts": {
            "total": num_fonts,
            "train": num_train_fonts,
            "ind_test": num_ind_test_fonts,
            "ood_test": num_ood_test_fonts,
        },
        "contents": {
            "total": num_contents,
            "train": num_train_contents,
            "ind_test": num_ind_test_contents,
            "ood_test": num_ood_test_contents,
        },
    }
    print("The number of fonts and contents in each split:")
    pprint(split_arg_dict)

    if num_train_fonts < 0:
        raise ValueError("num_train_fonts < 0")
    if num_train_contents < 0:
        raise ValueError("num_train_contents < 0")

    # sample generate split
    # sample_split_dict = {"total": 8, "train": 6, "ind_test": 3, "ood_test": 2}
    # generate_split(**sample_split_dict, verbose=True)

    # split fonts
    font_split_dict = generate_split_dict(**split_arg_dict["fonts"])

    font_split_df_dict = {}

    font_save_keys = ["filename", "name", "postScriptName"]
    for split, index_list in font_split_dict.items():
        font_split_df = None

        field_list_dict = defaultdict(list)
        for index in index_list:
            field_list_dict["index"].append(index)

            gfont_metadata = gfont_metadata_list[index]
            for font_save_key in font_save_keys:
                field_list_dict[font_save_key].append(gfont_metadata[font_save_key])

        font_split_df = pd.DataFrame(field_list_dict)
        font_split_df_dict[split] = font_split_df

    # split contents
    content_split_dict = generate_split_dict(**split_arg_dict["contents"])

    content_split_df_dict = {}

    for split, index_list in content_split_dict.items():
        content_split_df = None

        field_list_dict = defaultdict(list)
        for index in index_list:
            field_list_dict["index"].append(index)
            field_list_dict["content"].append(contents[index])

        # NOTE(xk): if index_list is empty, we make sure that the df has fields
        if not field_list_dict:
            field_list_dict["index"] = []
            field_list_dict["content"] = []

        content_split_df = pd.DataFrame(field_list_dict)
        content_split_df_dict[split] = content_split_df

    output_dir = Path(getattr(args, "output_dir"))
    output_dir.mkdir(exist_ok=True, parents=True)

    font_split_output_dir = output_dir / "font"
    font_split_output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving font split to {font_split_output_dir}")
    for split, df in font_split_df_dict.items():
        output_path = font_split_output_dir / f"{split}.tsv"
        if len(df) == 0:
            print(f"Empty df for {output_path}. Skipping.")
            continue
        df.to_csv(output_path, sep="\t", index=False)
        print(f"Saved {output_path}")

    content_split_output_dir = output_dir / "content"
    content_split_output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving content split to {content_split_output_dir}")
    for split, df in content_split_df_dict.items():
        output_path = content_split_output_dir / f"{split}.tsv"
        if len(df) == 0:
            print(f"Empty df for {output_path}. Skipping.")
            continue
        df.to_csv(output_path, sep="\t", index=False)
        print(f"Saved {output_path}")


def generate_split_dict(total, train, ind_test, ood_test, verbose=False):
    if total != train + ood_test:
        print(
            f"total != train + ood_test: {total} != {train} + {ood_test}. Not using all."
        )

    ood_test_index = np.random.choice(total, ood_test, replace=False)
    train_index = np.setdiff1d(np.arange(total), ood_test_index, assume_unique=True)
    ind_test_index = np.random.choice(train_index, ind_test, replace=False)

    if len(train_index) != train:
        train_index = np.random.choice(train_index, train, replace=False)

    return_dict = {
        "train": train_index.tolist(),
        "ind_test": ind_test_index.tolist(),
        "ood_test": ood_test_index.tolist(),
    }
    if verbose:
        print("======= print split indices =======")
        pprint(return_dict)
        print("======= print split indices =======")
    return return_dict


if __name__ == "__main__":
    main()
