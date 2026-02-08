"""
python src/svg_word_generation/tools/check_metadata_font_name_repeat.py \
    data/google_font_processor/google_font_metadata.filtered.jsonl
"""

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import click


NAME_FIELDS = [
    "filename",
    "font_dir_name",
    "fullName",
    "name",
    "postScriptName",
]


@click.command()
@click.argument(
    "input_metadata_jsonl_path",
    type=Path,
    default="data/google_font_processor/google_font_metadata.filtered.jsonl",
)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    input_metadata_jsonl_path = Path(args.input_metadata_jsonl_path)
    gfont_metadata_list = [
        json.loads(line) for line in input_metadata_jsonl_path.read_text().splitlines()
    ]

    for name_field in NAME_FIELDS:
        full_name_list = [x[name_field] for x in gfont_metadata_list]

        # Count the occurrences of each name
        name_counts = Counter(full_name_list)

        # Find the name(s) that appear more than once
        repeated_names = [
            (name, count) for name, count in name_counts.items() if count > 1
        ]
        repeated_names.sort(key=lambda x: x[1], reverse=True)
        print(f"Repeated names for {name_field}: {repeated_names}")


if __name__ == "__main__":
    main()
