"""
python src/svg_glyph_gen_v2/load_img_for_hf_dataset.py misc/250910-alphanumeric-abs_coord-ref_img-gemma3-tokenized

python src/svg_glyph_gen_v2/load_img_for_hf_dataset.py misc/250903-alphanumeric-ref_img-gemma3-tokenized
"""

import click
import datasets
from PIL import Image as PILImage


@click.command()
@click.argument(
    "input_dataset_dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    required=True,
)
def main(input_dataset_dir):
    # load
    dataset_dict = datasets.load_from_disk(input_dataset_dir)

    new_dataset = datasets.DatasetDict()
    for name, dataset in dataset_dict.items():
        dataset = dataset.map(to_pil)
        dataset = dataset.cast_column(
            "images", datasets.Sequence(datasets.Image(decode=True))
        )
        new_dataset[name] = dataset

    new_dataset_dir = input_dataset_dir + "-pil"
    new_dataset.save_to_disk(new_dataset_dir)
    print(f"Saved to {input_dataset_dir}")


def to_pil(example):
    example["images"] = [PILImage.open(i) for i in example["images"]]
    return example


if __name__ == "__main__":
    main()
