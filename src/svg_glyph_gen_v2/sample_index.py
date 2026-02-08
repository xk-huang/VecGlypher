"""
python src/svg_glyph_gen_v2/sample_index.py \
    data/processed/split_train_test_index/alphanumeric/font_family/train.tsv \
    100 \
    data/processed/split_train_test_index/alphanumeric/font_family/train-sample_100.tsv
"""

import click
import numpy as np
import pandas as pd


@click.command()
@click.argument("input_split_tsv", type=click.Path(exists=True))
@click.argument("num_samples", type=int, required=True)
@click.argument("output_split_tsv", type=click.Path(), required=True)
@click.option("--seed", type=int, default=42)
def main(input_split_tsv, num_samples, output_split_tsv, seed):
    print(f"Loading {input_split_tsv}")
    df = pd.read_csv(input_split_tsv, sep="\t")

    rng = np.random.default_rng(seed)
    df = df.sample(n=num_samples, random_state=rng)

    df.to_csv(output_split_tsv, sep="\t", index=False)
    print(f"Saved randomly-selected {num_samples} samples to {output_split_tsv}")


if __name__ == "__main__":
    main()
