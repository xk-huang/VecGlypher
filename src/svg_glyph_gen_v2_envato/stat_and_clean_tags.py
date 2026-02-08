"""
NOTE: not that work. Need to use LLM to split word, and filter unused tags.

python -m src.svg_glyph_gen_v2_envato.stat_and_clean_tags
"""

import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import click
import wordninja

from ..svg_glyph_gen_v2.utils import load_jsonl, setup_logger


@click.command()
@click.option(
    "--input_metadata_jsonl",
    type=click.Path(exists=True),
    default="data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram",
)
@click.option(
    "--output_dir",
    type=click.Path(),
    default="data/processed_envato/metadata_in_gfont-stat_and_clean_tags",
)
@click.option("--num_workers", type=int, default=20)
@click.option("--overwrite", type=bool, default=False)
def main(**kargs):
    args = SimpleNamespace(**kargs)

    font_filter = TagWordSplitter(args)
    font_filter.run()


class TagWordSplitter:
    def __init__(self, args):
        self.args = args

        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        self.logger = setup_logger(output_dir)

        # Initialize field value collections
        self.tag_set = set()
        self.key = "tags"
        self.total_num_skipped_tag = 0
        self.num_remained_tag_list = []
        self.num_tag_list = []

    def run(self):
        metadata = []
        input_metadata_jsonl = self.args.input_metadata_jsonl
        input_metadata_jsonl = Path(input_metadata_jsonl)
        if not input_metadata_jsonl.exists():
            raise FileNotFoundError(f"{input_metadata_jsonl} does not exist.")

        metadata = load_jsonl(input_metadata_jsonl)
        for metadata_item in metadata:
            self.parse_metadata_item(metadata_item)

        tag2split = {}
        for tag in self.tag_set:
            split_tag = self.split_word_in_tag(tag)
            tag2split[tag] = split_tag

        output_path = self.output_dir / "tag2split.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tag2split, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Saved {output_path}")
        self.logger.info(
            f"Skipped {self.total_num_skipped_tag} tags, total {len(self.tag_set)} tags"
        )
        self.logger.info(
            f"minimal number of remained tags: {min(self.num_remained_tag_list)}"
        )
        self.plot_num_remained_tag_to_bar()

    def plot_num_remained_tag_to_bar(self):
        import matplotlib.pyplot as plt

        num_remained_tag_to_bar = defaultdict(int)
        for num_remained_tag in self.num_remained_tag_list:
            num_remained_tag_to_bar[num_remained_tag] += 1
        # sort by key
        num_remained_tag_to_bar = dict(
            sorted(num_remained_tag_to_bar.items(), key=lambda x: x[0])
        )
        # plot
        plt.bar(
            list(num_remained_tag_to_bar.keys()),
            list(num_remained_tag_to_bar.values()),
        )
        plt.xlabel("Font")
        plt.ylabel("Number of remained tags")
        plt.title("Number of remained tags per font")
        # add value to each bar
        for i, v in enumerate(num_remained_tag_to_bar.values()):
            plt.text(i, v + 0.5, str(v), ha="center", va="bottom")

        output_path = self.output_dir / "num_remained_tag_to_bar.png"
        plt.savefig(output_path)
        self.logger.info(f"Saved {output_path}")

    def split_word_in_tag(self, tag):
        split_words = wordninja.split(tag)
        if len(split_words) == len(tag):
            return tag
        return " ".join(split_words)

    def parse_metadata_item(self, gfont_metadata):
        key = "tags"

        font_name = gfont_metadata["name"]
        font_name_set = font_name.split(" ")
        font_name_set = set(i.lower() for i in font_name_set)

        tag_list = gfont_metadata[key]
        num_tag = len(tag_list)
        self.num_tag_list.append(num_tag)

        num_remained_tag = 0
        for tag in tag_list:
            if tag.lower() in font_name_set:
                # self.logger.warning(
                #     f'Skip tag "{tag}" in "{font_name}", it is in font name, skip'
                # )
                self.total_num_skipped_tag += 1
                continue
            num_remained_tag += 1
            self.tag_set.add(tag)

        self.num_remained_tag_list.append(num_remained_tag)


if __name__ == "__main__":
    main()
