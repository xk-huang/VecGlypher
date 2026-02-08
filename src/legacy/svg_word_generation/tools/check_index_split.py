"""
python src/svg_word_generation/tools/check_index_split.py <index_split_tsv>
"""

from types import SimpleNamespace

import click
import pandas as pd


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    input_file = args.input_file

    df = pd.read_csv(input_file, sep="\t")

    print(f"input_file = {input_file}")
    print(f"len(df) = {len(df)}")
    print(df)


if __name__ == "__main__":
    main()
