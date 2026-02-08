"""
python src/tools/merge_hf_dataset.py \
    -i misc/250903-alphanumeric-gemma3-tokenized \
    -i misc/250903-alphanumeric-ref_img-gemma3-tokenized-pil \
    -o misc/250903-alphanumeric-text_img_merged-gemma3-tokenized-pil

python src/tools/merge_hf_dataset.py \
    -i misc/250910-alphanumeric-abs_coord-gemma3-tokenized \
    -i misc/250910-alphanumeric-abs_coord-ref_img-gemma3-tokenized-pil \
    -o misc/250910-alphanumeric-abs_coord-text_img_merged-gemma3-tokenized-pil
"""

from pathlib import Path

import click
import datasets


@click.command()
@click.option("--input_dataset_dir", "-i", multiple=True)
@click.option("--output_dataset_dir", "-o", required=True, type=str)
def main(input_dataset_dir, output_dataset_dir):
    output_dataset_dir = Path(output_dataset_dir)
    if output_dataset_dir.exists():
        raise ValueError(f"{output_dataset_dir} already exists, remove it first.")

    # Load the datasets
    print(f"Loading datasets from {input_dataset_dir}")
    input_datasets = []
    for input_dataset_dir_ in input_dataset_dir:
        dataset = datasets.load_from_disk(input_dataset_dir_)
        input_datasets.append(dataset)

    # Merge the datasets
    merged_dataset = datasets.DatasetDict()
    # make sure keys are the same
    for key in input_datasets[0].keys():
        print(f"Merging {key}")
        merged_dataset[key] = datasets.concatenate_datasets(
            [input_dataset[key] for input_dataset in input_datasets]
        )
    print(f"input datasets: {input_datasets}")
    print(f"merged dataset: {merged_dataset}")

    # Save the merged dataset
    merged_dataset.save_to_disk(output_dataset_dir)
    print(f"Saved merged dataset to {output_dataset_dir}")


if __name__ == "__main__":
    main()
