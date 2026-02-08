"""
Split train and val and outputs the index tsv file

Support 3 types of input:
- font: split by font (deprecated, there are font family with +9 weights, which might cause train/test leakage)
- font_family: split by font family
- content: split by content

python -m src.svg_glyph_gen_v2.split_train_test_index_v2 --input_type font_family --num_ind_test 150 --num_ood_test 150
# python -m src.svg_glyph_gen_v2.split_train_test_index_v2 --input_type font --num_ind_test 250 --num_ood_test 250
python -m src.svg_glyph_gen_v2.split_train_test_index_v2 --input_type content --num_ind_test 10 --num_ood_test 0
"""

import json
from collections import defaultdict
from pathlib import Path

from pprint import pformat, pprint
from types import SimpleNamespace

import click
import numpy as np
import pandas as pd

from .utils import load_jsonl, setup_logger


@click.command()
@click.option(
    "--gfont_metadata_path",
    # default="data/processed/google_font_metadata-filter_invalid-filter_by_pangram",
    required=True,
)
@click.option(
    "--content_path",
    # default="data/processed/content/alphanumeric.txt",
    required=True,
)
@click.option(
    "--input_type",
    required=True,
    type=click.Choice(["font", "font_family", "content"]),
)
@click.option("--num_train", default=None, type=int)
@click.option("--num_ind_test", default=100, type=int)
@click.option("--num_ood_test", default=100, type=int)
@click.option("--num_dev", default=10, type=int)
@click.option(
    "--output_dir",
    # default="data/processed/split_train_test_index/alphanumeric",
    required=True,
)
@click.option("--seed", default=42, type=int)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    train_test_index_splitter = TrainTestIndexSplitter(args)

    train_test_index_splitter.run()


class TrainTestIndexSplitter:
    def __init__(self, args):
        self.args = args
        output_dir = Path(args.output_dir)
        output_dir = output_dir / f"{args.input_type}"
        self.logger = setup_logger(output_dir)
        self.output_dir = output_dir
        self.logger.info(f"Output dir: {self.output_dir}")

        seed = args.seed
        np.random.seed(seed)
        self.logger.info(f"Set random seed to {seed}")

    def run(self):
        args = self.args
        data = self.load_data()

        if args.input_type == "font_family":
            split_df_dict = self.split_font_family_train_test_index(data)
        elif args.input_type == "font":
            raise DeprecationWarning(
                f"there are font family with +9 weights, which might cause train/test leakage"
            )
            split_df_dict = self.split_font_train_test_index(data)
        elif args.input_type == "content":
            split_df_dict = self.split_content_train_test_index(data)
        else:
            raise NotImplementedError(f"Unknown input type {args.input_type}")
        self.logger.info(f"input_type: {args.input_type}")

        for split, df in split_df_dict.items():
            output_path = self.output_dir / f"{split}.tsv"
            if len(df) == 0:
                print(f"Empty df for {output_path}. Skipping.")
                continue
            df.to_csv(output_path, sep="\t", index=False)
            print(f"Saved {output_path}")

    def prepare_split_arg(self, data):
        args = self.args
        # prepare split arg
        num_train = args.num_train
        num_ind_test = args.num_ind_test
        num_ood_test = args.num_ood_test
        num_dev = args.num_dev
        num_data = len(data)

        if num_train is None:
            num_train = num_data - num_ood_test
        else:
            num_left = num_data - num_ood_test
            if num_train > num_left:
                err_msg = f"num_train > num_data: {num_train} > {num_left}"
                self.logger.error(err_msg)
                raise ValueError(err_msg)
        if num_train < 0:
            err_msg = f"num_train < 0: {num_train} < 0"
            self.logger.error(err_msg)
            raise ValueError(err_msg)

        split_arg = {
            "total": num_data,
            "train": num_train,
            "ind_test": num_ind_test,
            "ood_test": num_ood_test,
            "dev": num_dev,
        }
        return split_arg

    def split_content_train_test_index(self, data):
        """
        split the data into train, ind_test, ood_test, dev
        """
        # prepare split arg for font families
        split_arg = self.prepare_split_arg(data)
        self.logger.info(f"split_arg: {split_arg}")
        split_dict = self._generate_split_dict(**split_arg, verbose=False)

        content_split_df_dict = {}

        for split, index_list in split_dict.items():
            content_split_df = None

            field_list_dict = defaultdict(list)
            for index in index_list:
                field_list_dict["index"].append(index)
                field_list_dict["content"].append(data[index])

            # NOTE(xk): if index_list is empty, we make sure that the df has fields
            if not field_list_dict:
                field_list_dict["index"] = []
                field_list_dict["content"] = []

            content_split_df = pd.DataFrame(field_list_dict)
            content_split_df_dict[split] = content_split_df

        return content_split_df_dict

    # FONT_SAVE_KEYS = ["filename", "postScriptName", "name", "font_family_dir_name"]
    FONT_SAVE_KEYS = ["filename", "font_family_dir_name"]

    def split_font_train_test_index(self, data):
        """
        split the data into train, ind_test, ood_test, dev
        """
        # prepare split arg for font families
        split_arg = self.prepare_split_arg(data)
        self.logger.info(f"split_arg: {split_arg}")
        split_dict = self._generate_split_dict(**split_arg, verbose=False)

        font_split_df_dict = {}

        for split, index_list in split_dict.items():
            font_split_df = None

            field_list_dict = defaultdict(list)
            for index in index_list:
                field_list_dict["index"].append(index)

                gfont_metadata = data[index]
                for font_save_key in self.FONT_SAVE_KEYS:
                    field_list_dict[font_save_key].append(gfont_metadata[font_save_key])

            font_split_df = pd.DataFrame(field_list_dict)
            font_split_df_dict[split] = font_split_df

        return font_split_df_dict

    def split_font_family_train_test_index(self, data):
        """
        each font family may have multiple fonts.
        We first split the font families into train, ind_test, ood_test, dev.
        And then distribute the fonts to the corresponding split.
        """
        from collections import Counter, defaultdict

        font_family_list = [d["font_family_dir_name"] for d in data]

        font_family_counter = Counter(font_family_list)
        self.plot_font_family_distribution(font_family_counter, 0.2)

        unique_font_family_list = list(font_family_counter.keys())

        # prepare split arg for font families
        split_arg = self.prepare_split_arg(unique_font_family_list)
        self.logger.info(f"split_arg for font family: {split_arg}")
        split_dict = self._generate_split_dict(**split_arg, verbose=False)

        # build split index mapping from font family to data index
        font_family2font_family_index = {
            k: i for i, k in enumerate(unique_font_family_list)
        }
        font_family_index2data_index = defaultdict(list)
        for data_index, d in enumerate(data):
            font_family = d["font_family_dir_name"]
            font_family_index = font_family2font_family_index[font_family]
            font_family_index2data_index[font_family_index].append(data_index)

        font_split_df_dict = {}

        for split, font_family_index_list in split_dict.items():
            font_split_df = None

            field_list_dict = defaultdict(list)
            for font_family_index in font_family_index_list:
                data_index_list = font_family_index2data_index[font_family_index]
                for data_index in data_index_list:
                    field_list_dict["index"].append(data_index)
                    gfont_metadata = data[data_index]
                    for font_save_key in self.FONT_SAVE_KEYS:
                        field_list_dict[font_save_key].append(
                            gfont_metadata[font_save_key]
                        )

            font_split_df = pd.DataFrame(field_list_dict)
            font_split_df_dict[split] = font_split_df

        stats_dict = {}
        for split, df in font_split_df_dict.items():
            stats_dict[split] = len(df)
        self.logger.info(f"stats_dict: {stats_dict}")
        self._validate_no_overlap_font_family(font_split_df_dict)

        # plot ood

        font_family_counter = Counter(
            font_split_df_dict["ood_test"]["font_family_dir_name"]
        )
        self.plot_font_family_distribution(font_family_counter, 1.0, "-ood_test")

        return font_split_df_dict

    def _validate_no_overlap_font_family(self, font_split_df_dict):
        split_set = {}
        for split, df in font_split_df_dict.items():
            split_set[split] = set(df["font_family_dir_name"].tolist())
        # validate train and ood_test don't have overlap
        if len(split_set["train"] & split_set["ood_test"]) > 0:
            err_msg = f"train and ood_test have overlap: {split_set['train'] & split_set['ood_test']}"
            self.logger.error(err_msg)
            raise ValueError(err_msg)
        # validate ind_test/dev is in train
        if len(split_set["ind_test"] - split_set["train"]) > 0:
            err_msg = f"ind_test is not in train: {split_set['ind_test'] - split_set['train']}"
            self.logger.error(err_msg)
            raise ValueError(err_msg)
        if len(split_set["dev"] - split_set["train"]) > 0:
            err_msg = f"dev is not in train: {split_set['dev'] - split_set['train']}"
            self.logger.error(err_msg)
        self.logger.info("No overlap between train and ood_test")
        self.logger.info("ind_test and dev are in train")

    def _generate_split_dict(
        self, total, train, ind_test, ood_test, dev, verbose=False
    ):
        if total != train + ood_test:
            print(
                f"total != train + ood_test: {total} != {train} + {ood_test}. Not using all."
            )

        ood_test_index = np.random.choice(total, ood_test, replace=False)
        train_index = np.setdiff1d(np.arange(total), ood_test_index, assume_unique=True)
        ind_test_index = np.random.choice(train_index, ind_test, replace=False)
        dev_index = np.random.choice(train_index, dev, replace=False)

        if len(train_index) != train:
            train_index = np.random.choice(train_index, train, replace=False)

        return_dict = {
            "train": train_index.tolist(),
            "ind_test": ind_test_index.tolist(),
            "ood_test": ood_test_index.tolist(),
            "dev": dev_index.tolist(),
        }
        if verbose:
            self.logger.info(
                "======= print split indices ======\n"
                f"{return_dict}\n"
                "======= print split indices ======="
            )
        return return_dict

    def plot_font_family_distribution(
        self, font_family_counter, top_percent=None, name_suffix=""
    ):
        import matplotlib.pyplot as plt

        # plot bar chart for top 10% font families
        num_families = len(font_family_counter)
        if top_percent is None:
            top_percent = 1.0
        top_n = max(1, int(num_families * top_percent))
        top_n_families = font_family_counter.most_common(top_n)
        font_names, font_counts = zip(*top_n_families)

        plt.figure(figsize=(12, 6))
        plt.bar(font_names, font_counts)
        plt.xticks([])
        plt.title(f"Top {top_n} Font Family Distribution")
        plt.xlabel("Font Family")
        plt.ylabel("Count")
        # plt.xticks(rotation=90, ha="center")  # Make bar names vertical
        plt.tight_layout()
        plot_path = (
            self.output_dir / f"top{top_n}_font_family_distribution{name_suffix}.png"
        )
        plt.savefig(plot_path)
        self.logger.info(
            f"Saved top {top_n} font family distribution plot to {plot_path}"
        )
        plt.close()

    def load_data(self):
        input_type = self.args.input_type
        if input_type == "font" or input_type == "font_family":
            gfont_metadata_path = Path(self.args.gfont_metadata_path)
            gfont_metadata_list = load_jsonl(gfont_metadata_path, logger=self.logger)
            return gfont_metadata_list
        elif input_type == "content":
            content_path = self.args.content_path
            contents = []
            with open(content_path, "r") as f:
                for line in f.readlines():
                    contents.append(line.strip("\n"))

            self.logger.info(f"Loaded {len(contents)} content from {content_path}")
            return contents
        else:
            self.logger.error(f"Unknown input type {input_type}")
            raise ValueError(f"Unknown input type {input_type}")


if __name__ == "__main__":
    main()
