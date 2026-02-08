"""
python src/tools/stats_dataset_token_length.py data/processed/sft/250814-oxford_5000-100_fonts-apply_word_sep/train-sample_100 misc/stats_dataset_token_length/250814-oxford_5000-100_fonts-apply_word_sep-train-sample_100
python src/tools/stats_dataset_token_length.py data/processed/sft/250814-oxford_5000-100_fonts-apply_word_sep/train-alphanumeric misc/stats_dataset_token_length/250814-oxford_5000-100_fonts-apply_word_sep-train-alphanumeric
python src/tools/stats_dataset_token_length.py data/processed/sft/250813-alphanumeric/train_font_family misc/stats_dataset_token_length/250813-alphanumeric-train_font_family
"""

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt


@click.command()
@click.argument(
    "input_dir",
    default="data/processed/sft/250813-oxford_5000-100_fonts/train-sample_100",
    # help="Input directory containing JSONL files",
)
@click.argument(
    "output_dir",
    default="misc/stats_dataset_token_length",
    # help="Output directory for results",
)
def main(input_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_plot_path = output_dir / "token_length.png"

    input_jsonl_list = list(Path(input_dir).glob("*.jsonl"))

    token_length_list = []
    for input_jsonl in input_jsonl_list:
        with open(input_jsonl, "r") as f:
            for line in f:
                metadata = json.loads(line)
                token_length_list.append(len(metadata["output"]))

    # plot hist
    plt.hist(token_length_list, bins=100)
    plt.xlabel("token length")
    plt.ylabel("count")
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Saved token length plot to {output_plot_path}")

    # save to json
    output_json_path = output_dir / "token_length.json"
    with open(output_json_path, "w") as f:
        json.dump(token_length_list, f, indent=4)
    print(f"Saved token length to {output_json_path}")


if __name__ == "__main__":
    main()
