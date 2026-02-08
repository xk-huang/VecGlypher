"""
python src/tools/count_lines.py [FILENAME_OR_DIR]
"""

import os
from pathlib import Path

import click


def _count_lines(filename):
    count = 0
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            count += chunk.count(b"\n")
    return count


def count_lines(filename_or_dir):
    filename_or_dir = Path(filename_or_dir)
    if not filename_or_dir.exists():
        raise FileNotFoundError(f"Path not found: {filename_or_dir}")

    if os.path.isfile(filename_or_dir):
        return _count_lines(filename_or_dir)
    else:
        return sum(_count_lines(f) for f in Path(filename_or_dir).glob("*.jsonl"))


@click.command()
@click.argument("filename_or_dir")
def main(filename_or_dir):
    print(count_lines(filename_or_dir))


if __name__ == "__main__":
    main()
