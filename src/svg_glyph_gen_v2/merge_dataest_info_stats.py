"""
python src/svg_glyph_gen_v2/merge_dataest_info_stats.py data/processed_envato/filtered_sft
"""

import json
from pathlib import Path

import click


@click.command()
@click.argument(
    "input_dataset_dir",
    # default="data/processed/sft/250813-alphanumeric/",
    required=True,
)
def main(input_dataset_dir):
    input_dataset_dir = Path(input_dataset_dir)

    merged_dataset_info_path = input_dataset_dir / "dataset_info.json"
    merged_dataset_stat_path = input_dataset_dir / "dataset_stat.json"
    if merged_dataset_info_path.exists():
        merged_dataset_info_path.unlink()
        print(f"Deleted existing merged dataset info at {merged_dataset_info_path}")
    if merged_dataset_stat_path.exists():
        merged_dataset_stat_path.unlink()
        print(f"Deleted existing merged dataset info at {merged_dataset_info_path}")

    dataset_info_path_list = list(input_dataset_dir.glob("**/dataset_info.json"))

    merged_dataset_info = {}
    merged_dataset_stat = {}
    for dataset_info_path in dataset_info_path_list:
        dataset_dir = dataset_info_path.parent
        dataset_dir_name = dataset_dir.name

        with open(dataset_info_path, "r") as f:
            data = json.load(f)

        for subset_name, subset_dict in data.items():
            subset_dict = subset_dict.copy()
            file_name = subset_dict["file_name"]
            if file_name != subset_name:
                raise ValueError(f"file_name {file_name} != subset_name {subset_name}")

            updated_subset_name = f"{dataset_dir_name}/{subset_name}"
            subset_dict["file_name"] = updated_subset_name
            merged_dataset_info[updated_subset_name] = subset_dict

        # process dataset_stat.json
        dataset_stat_path = dataset_dir / "dataset_stat.json"
        with open(dataset_stat_path, "r") as f:
            data = json.load(f)
        for subset_name, subset_dict in data.items():
            updated_subset_name = f"{dataset_dir_name}/{subset_name}"
            merged_dataset_stat[updated_subset_name] = subset_dict

    with open(merged_dataset_info_path, "w") as f:
        json.dump(merged_dataset_info, f, indent=4)
    print(f"Saved merged dataset info to {merged_dataset_info_path}")

    with open(merged_dataset_stat_path, "w") as f:
        json.dump(merged_dataset_stat, f, indent=4)
    print(f"Saved merged dataset info to {merged_dataset_stat_path}")


if __name__ == "__main__":
    main()
