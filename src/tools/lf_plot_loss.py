"""
python src/tools/lf_plot_loss.py -i <input-dir> [-k <keys>]
"""

from pathlib import Path

import click
from llamafactory.extras.ploting import plot_loss


@click.command()
@click.option("--input-dir", "-i", type=Path, required=True)
@click.option("--keys", "-k", type=str, default="loss,eval_loss,epoch")
def main(input_dir: Path, keys):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise ValueError(f"input-dir {input_dir} does not exist")

    print(f"input-dir {input_dir}")
    keys = keys.split(",")
    print(f"keys {keys}")
    plot_loss(input_dir, keys)


if __name__ == "__main__":
    main()
