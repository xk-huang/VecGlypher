"""
python src/svg_glyph_gen_v2/build_dataset_info.py data/processed/sft/250813-alphanumeric/
"""

import json
from pathlib import Path

import click


@click.command()
@click.argument(
    "dataset_dir",
    # default="data/processed/sft/250813-alphanumeric/",
    required=True,
)
@click.option("--add_images", is_flag=True, default=False)
def main(dataset_dir, add_images):
    dataset_dir = Path(dataset_dir)

    # all sub datasets are uner the dataset_dir
    sub_dataset_dirs = [i for i in dataset_dir.glob("*") if i.is_dir()]
    sub_dataset_rel_dirs = [x.relative_to(dataset_dir) for x in sub_dataset_dirs]
    sub_dataest_names = [x.stem for x in sub_dataset_rel_dirs]

    dataset_info_item = {
        "file_name": "/mnt/workspace/svg_glyph_llm/data/processed/sft/alphanumeric-train_fonts",
        "formatting": "alpaca",
        "columns": {
            "system": "system",
            "prompt": "instruction",
            "response": "output",
        },
    }
    dataset_info = {}
    for sub_dataset_dir, sub_dataset_rel_dir, sub_dataset_name in zip(
        sub_dataset_dirs, sub_dataset_rel_dirs, sub_dataest_names
    ):
        if len(list(sub_dataset_dir.glob("*.jsonl"))) == 0:
            continue
        dataset_info_item_ = dataset_info_item.copy()
        if add_images:
            dataset_info_item_["columns"].update({"images": "images"})
        dataset_info_item_["file_name"] = str(sub_dataset_rel_dir)
        dataset_info[str(sub_dataset_name)] = dataset_info_item_

    dataset_json_path = dataset_dir / "dataset_info.json"
    with open(dataset_json_path, "w") as f:
        json.dump(dataset_info, f, indent=4)
    print(f"dataset info saved to {dataset_json_path}")


if __name__ == "__main__":
    main()
