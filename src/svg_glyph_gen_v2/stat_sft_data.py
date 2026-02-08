"""
python src/svg_glyph_gen_v2/stat_sft_data.py data/processed/filtered_sft/250813-alphanumeric
"""

import json
from collections import OrderedDict

from pathlib import Path
from types import SimpleNamespace

import click
import tqdm


def _count_jsonl_lines(jsonl_file_path: Path):
    num_samples = 0
    num_failed_samples = 0

    with open(jsonl_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            try:
                line = json.loads(line)
                num_samples += 1
            except Exception:
                num_failed_samples += 1
    return {"num_samples": num_samples, "num_failed_samples": num_failed_samples}


def count_jsonl_samples(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist.")

    if input_path.is_file():
        return _count_jsonl_lines(input_path)

    result_dict = {"num_samples": 0, "num_failed_samples": 0}
    jsonl_list = list(input_path.glob("*.jsonl"))
    print(f"Found {len(jsonl_list)} jsonl files in {input_path}")
    if len(jsonl_list) == 0:
        return None
    for jsonl_file_path in tqdm.tqdm(jsonl_list):
        result = _count_jsonl_lines(jsonl_file_path)
        result_dict["num_samples"] += result["num_samples"]
        result_dict["num_failed_samples"] += result["num_failed_samples"]

    return result_dict


@click.command()
@click.argument("input_sft_data_dir", type=click.Path(exists=True))
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    input_sft_data_dir = Path(args.input_sft_data_dir)
    dataset_info_json_path = input_sft_data_dir / "dataset_info.json"
    with open(dataset_info_json_path, "r") as f:
        dataset_info = json.load(f)

    stat_dict = {}
    for split_name in dataset_info:
        split_info = dataset_info[split_name]
        file_name = split_info["file_name"]
        split_dir = input_sft_data_dir / file_name
        result_stat = count_jsonl_samples(split_dir)
        if result_stat is not None:
            stat_dict[split_name] = result_stat

    output_stat_json_path = input_sft_data_dir / "dataset_stat.json"
    with open(output_stat_json_path, "w") as f:
        stat_dict = OrderedDict(sorted(stat_dict.items()))
        json.dump(stat_dict, f, indent=4)
    print(f"Saved dataset stat to {output_stat_json_path}")


if __name__ == "__main__":
    main()
