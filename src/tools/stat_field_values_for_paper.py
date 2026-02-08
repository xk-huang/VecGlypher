"""
python -m src.tools.stat_field_values_for_paper \
--input_metadata_jsonl data/processed/google_font_metadata-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr \
--output_dir misc/stat_field_values_for_paper/google_fonts \
--overwrite --split_words --max_font_size 120

python -m src.tools.stat_field_values_for_paper \
--input_metadata_jsonl data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr \
--output_dir misc/stat_field_values_for_paper/envato_fonts \
--overwrite --max_font_size 120  --min_font_size 10

tar -czf misc/stat_field_values_for_paper.tar.gz $(find misc/stat_field_values_for_paper -type f -name "*.pdf" -o -name "*.svg")
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import List

import click
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.terminal.embed import embed
from wordcloud import WordCloud

from ..svg_glyph_gen_v2.utils import load_jsonl, prepare_output_dir_and_logger

plt.rcParams.update(
    {
        "font.family": "serif",
    }
)
sns.set_style("ticks")
sns.set_context(
    "paper",
    font_scale=2,
)
sns.set_palette("pastel")


@click.command()
@click.option(
    "--input_metadata_jsonl",
    type=click.Path(exists=True),
    # default="data/processed/google_font_metadata-filter_invalid-filter_by_pangram",
    required=True,
)
@click.option(
    "--output_dir",
    type=click.Path(),
    # default="data/processed/google_font_metadata-stat_field_values",
    required=True,
)
@click.option("--num_workers", type=int, default=20)
@click.option("--max_font_size", type=int, default=None)
@click.option("--min_font_size", type=int, default=4)
@click.option("--max_words", type=int, default=None)
@click.option("--seed", type=int, default=1)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--split_words", is_flag=True, default=False)
def main(**kargs):
    args = SimpleNamespace(**kargs)

    font_filter = Stats(args)
    font_filter.run()


class Stats:
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

        # Initialize field value collections
        self.field_values = defaultdict(lambda: defaultdict(int))

    def run(self):
        metadata = []
        input_metadata_jsonl = self.args.input_metadata_jsonl
        input_metadata_jsonl = Path(input_metadata_jsonl)
        if not input_metadata_jsonl.exists():
            raise FileNotFoundError(f"{input_metadata_jsonl} does not exist.")

        metadata = load_jsonl(input_metadata_jsonl)
        for metadata_item in metadata:
            self.parse_metadata_item(metadata_item)

        # Output statistics
        self.output_statistics()

    KEYS = [
        "filename",
        "font_family_dir_name",
        "name",
        "tags",
        "source",
        "category",
        # only google font
        "fullName",
        "tag_weights",
        "style",
        "weight",
        "num_fonts",
        "classifications",
        "stroke"
        # only envato font
        "spacings",
        "identifier",
    ]

    def parse_metadata_item(self, gfont_metadata):
        for key in self.KEYS:
            if key in gfont_metadata:
                value = gfont_metadata[key]
                if isinstance(value, List):
                    for v in value:
                        self.field_values[key][v] += 1
                else:
                    self.field_values[key][value] += 1
        if "tags" in gfont_metadata:
            self.field_values["num_tags"][len(gfont_metadata["tags"])] += 1

    def output_statistics(self):
        """Output statistics for all field values"""
        self.logger.info("Field value statistics:")

        # for field_name, value_counts in self.field_values.items():
        #     # Sort by count (descending) then by value
        #     sorted_items = sorted(value_counts.items(), key=lambda x: (-x[1], x[0]))
        #     self.logger.info(f"\n{field_name} ({len(value_counts)} unique values):")
        #     for value, count in sorted_items:
        #         self.logger.info(f"  {value}: {count}")

        # Save statistics to JSON file
        stats_output = {}
        for field_name, value_counts in self.field_values.items():
            # Sort by count (descending) then by value
            sorted_items = sorted(value_counts.items(), key=lambda x: (-x[1], x[0]))
            stats_output[field_name] = {
                "unique_count": len(value_counts),
                "total_occurrences": sum(value_counts.values()),
                "values": [
                    {"value": value, "count": count} for value, count in sorted_items
                ],
            }

        output_file = self.output_dir / "field_value_statistics.json"
        with open(output_file, "w") as f:
            json.dump(stats_output, f, indent=2, default=str)

        # self.logger.info(f"\nStatistics saved to: {output_file}")

        if "num_tags" in stats_output:
            self.plot_distribution(stats_output["num_tags"], xlabel="Number of Tags")

        # Plot word cloud
        # stats_output["tags"], each dict contains two fields: {"value": str, "count": int},
        # "value" is the word, and count is the number of count of that word.
        tags_data = stats_output["tags"]["values"]
        if self.args.split_words:
            self.logger.info("Splitting words by space")
            freq_dict = {}
            for item in tags_data:
                if item["value"] in freq_dict:
                    freq_dict[item["value"]] += item["count"]
                else:
                    freq_dict[item["value"]] = item["count"]
                for word in re.split(r"[-\s]+", item["value"]):
                    if word in freq_dict:
                        freq_dict[word] += item["count"]
                    else:
                        freq_dict[word] = item["count"]
        else:
            freq_dict = {item["value"]: item["count"] for item in tags_data}
        wordcloud = WordCloud(
            width=1000,
            height=650,
            background_color="white",
            colormap="viridis",
            max_words=self.args.max_words,
            min_font_size=self.args.min_font_size,
            max_font_size=self.args.max_font_size,
            random_state=self.args.seed,  # reproducibility
        ).generate_from_frequencies(freq_dict)
        wordcloud_output_path = self.output_dir / "wordcloud.svg"
        with open(wordcloud_output_path, "w") as f:
            f.write(wordcloud.to_svg(embed_font=True))
        self.logger.info(f"Word cloud saved to: {wordcloud_output_path}")

        # Create plots for tags field
        # if "tags" in stats_output:
        #     # Plot top 20 tags
        #     self.plot_tags_distribution(stats_output["tags"], top_k=20)
        #     # Plot all tags
        #     # self.plot_tags_distribution(stats_output["tags"], top_k=None)

        #     tags_data = stats_output["tags"]
        #     values = []
        #     non_uniq_total_occurrences = 0
        #     uniq_values = []
        #     for item in tags_data["values"]:
        #         count = item["count"]
        #         if count > 1:
        #             values.append(item)
        #             non_uniq_total_occurrences += count
        #         else:
        #             uniq_values.append(item["value"])
        #     self.plot_tags_distribution(
        #         {
        #             "values": values,
        #             "unique_count": len(values),
        #             "total_occurrences": non_uniq_total_occurrences,
        #         },
        #         top_k=None,
        #     )
        #     uniq_tags_path = self.output_dir / "uniq_tags.txt"
        #     with open(uniq_tags_path, "w") as f:
        #         f.write("\n".join(uniq_values))

    def plot_distribution(
        self,
        tags_data,
        figsize=(8, 6),
        rotate_x=0,
        force_scientific=False,
        show_values=True,
        xlabel="Value",
        ylabel="Count",
    ):
        values = tags_data["values"]
        # sort by "counts"
        values = sorted(values, key=lambda x: x["value"], reverse=False)
        # get labels and counts
        labels = [p["value"] for p in values]
        counts = [p["count"] for p in values]

        fig, ax = plt.subplots(figsize=figsize)
        # ax.bar(range(len(labels)), counts, color="steelblue", alpha=0.7)
        sns.barplot(
            x=labels,
            y=counts,
            ax=ax,
        )
        sns.despine()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if show_values:
            for i, y in enumerate(counts):
                s = f"{y:.0e}" if force_scientific else f"{y:g}"
                ax.text(i, y, s, ha="center", va="bottom", fontsize=9)

        plot_file = self.output_dir / f"distrubution.pdf"
        fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        self.logger.info(f"distribution plot saved to: {plot_file}")


if __name__ == "__main__":
    main()
