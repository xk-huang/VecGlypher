"""
python -m src.svg_glyph_gen_v2.stat_field_values
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import List

import click
import matplotlib.pyplot as plt

from .utils import load_jsonl, prepare_output_dir_and_logger


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
@click.option("--overwrite", is_flag=True, default=False)
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

        self.logger.info(f"\nStatistics saved to: {output_file}")

        if "num_tags" in stats_output:
            self.plot_tags_distribution(
                stats_output["num_tags"],
                top_k=None,
                plot_title="num_tags",
                sort_by="value",
                reverse=False,
            )
        # Create plots for tags field
        if "tags" in stats_output:
            # Plot top 20 tags
            self.plot_tags_distribution(stats_output["tags"], top_k=20)
            # Plot all tags
            # self.plot_tags_distribution(stats_output["tags"], top_k=None)

            tags_data = stats_output["tags"]
            values = []
            non_uniq_total_occurrences = 0
            uniq_values = []
            for item in tags_data["values"]:
                count = item["count"]
                if count > 1:
                    values.append(item)
                    non_uniq_total_occurrences += count
                else:
                    uniq_values.append(item["value"])
            self.plot_tags_distribution(
                {
                    "values": values,
                    "unique_count": len(values),
                    "total_occurrences": non_uniq_total_occurrences,
                },
                top_k=None,
            )
            uniq_tags_path = self.output_dir / "uniq_tags.txt"
            with open(uniq_tags_path, "w") as f:
                f.write("\n".join(uniq_values))

    def plot_tags_distribution(
        self, tags_data, top_k=None, plot_title="tags", sort_by="count", reverse=True
    ):
        values = tags_data["values"]

        if top_k is None:
            selected_values = values
            plot_suffix = "all"
            title_prefix = "All"
        else:
            selected_values = values[:top_k]
            plot_suffix = f"top_{top_k}"
            title_prefix = f"Top {len(selected_values)}"

        selected_values = sorted(
            selected_values, key=lambda x: x[sort_by], reverse=reverse
        )
        tag_names = [item["value"] for item in selected_values]
        counts = [item["count"] for item in selected_values]

        # --- dynamic sizing ---
        if top_k is None:
            fig_height = max(8, len(tag_names) * 0.15)
            fontsize_yticks = 8
            fontsize_labels = 7
        else:
            fig_height = 8
            fontsize_yticks = 10
            fontsize_labels = 9

        # width from longest label + count digits (prevents extra whitespace)
        max_label_len = []
        for s in tag_names:
            if isinstance(s, str):
                max_label_len.append(len(s))
            else:
                max_label_len.append(len(str(s)))
        max_label_len = max(max_label_len, default=0)
        max_count = max(counts) if counts else 0
        digits = int(math.log10(max_count)) + 1 if max_count > 0 else 1
        fig_width = min(18, max(8, 6 + 0.12 * max_label_len + 0.5 * digits))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        self.logger.info(
            f"Creating {plot_title} distribution plot for {plot_suffix} {plot_title}"
        )
        bars = ax.barh(range(len(tag_names)), counts, color="steelblue", alpha=0.7)

        ax.set_yticks(range(len(tag_names)))
        ax.set_yticklabels(tag_names, fontsize=fontsize_yticks)
        ax.set_xlabel("Count", fontsize=12)
        title = (
            f"{title_prefix} {plot_title} Distribution\n"
            f'(Total: {tags_data["unique_count"]} unique {plot_title}, '
            f'{tags_data["total_occurrences"]} occurrences)'
        )
        ax.set_title(title, fontsize=14, pad=20)

        # --- eliminate huge blank above/below bars ---
        ax.margins(y=0)  # remove default 5% y margin
        ax.set_ylim(-0.5, len(tag_names) - 0.5)  # make limits snug to bars

        # count labels
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + max_count * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                ha="left",
                va="center",
                fontsize=fontsize_labels,
            )

        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        # tighter layout but leave room for the title
        plt.tight_layout()

        plot_file = self.output_dir / f"{plot_title}_distribution_{plot_suffix}.pdf"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        self.logger.info(f"{plot_title} distribution plot saved to: {plot_file}")
        plt.close()


if __name__ == "__main__":
    main()
