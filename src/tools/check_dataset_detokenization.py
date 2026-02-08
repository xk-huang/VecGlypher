"""
python src/tools/load_data_and_detokenize.py \
    -i misc/250903-alphanumeric-gemma3-tokenized \
    -m /home/vecglypher/mnt/workspace/hf_downloads/google/gemma-3-4b-it

python src/tools/load_data_and_detokenize.py \
    -i misc/250903-alphanumeric-ref_img-gemma3-tokenized-pil \
    -m /home/vecglypher/mnt/workspace/hf_downloads/google/gemma-3-4b-it

python src/tools/load_data_and_detokenize.py \
    -i misc/250903-alphanumeric-text_img_merged-gemma3-tokenized-pil \
    -m /home/vecglypher/mnt/workspace/hf_downloads/google/gemma-3-4b-it


python src/tools/load_data_and_detokenize.py \
    -i misc/250910-alphanumeric-abs_coord-gemma3-tokenized \
    -m /home/vecglypher/mnt/workspace/hf_downloads/google/gemma-3-4b-it

python src/tools/load_data_and_detokenize.py \
    -i misc/250910-alphanumeric-abs_coord-ref_img-gemma3-tokenized-pil \
    -m /home/vecglypher/mnt/workspace/hf_downloads/google/gemma-3-4b-it

python src/tools/load_data_and_detokenize.py \
    -i misc/250910-alphanumeric-abs_coord-text_img_merged-gemma3-tokenized-pil \
    -m /home/vecglypher/mnt/workspace/hf_downloads/google/gemma-3-4b-it
"""

from importlib.metadata import requires
from pathlib import Path

import click
import datasets
import torch
from transformers import AutoTokenizer


@click.command()
@click.option("--input_dataset_dir", "-i", required=True)
@click.option("--model_path", "-m", required=True)
def main(input_dataset_dir, model_path):
    # Load the datasets
    print(f"Loading datasets from {input_dataset_dir}")
    dataset = datasets.load_from_disk(input_dataset_dir)

    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    split2sample = {}
    for split_name, split in dataset.items():
        sample = split[0]
        split2sample[split_name] = sample

    for split_name, sample in split2sample.items():
        print("=================== Start of Sample ==================")
        print(f"Processing {split_name}")
        # print(f"Sample: {sample}")

        input_ids = sample["input_ids"]
        labels = sample["labels"]

        decoded_input_ids = tokenizer.decode(
            input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        labels = torch.tensor(labels)
        labels = labels.masked_fill(labels == -100, tokenizer.pad_token_id)
        decoded_labels = tokenizer.decode(
            labels, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        print(f"decoded_input_ids:\n{decoded_input_ids}")
        print(f"decoded_labels:\n{decoded_labels}")

        print("=================== End of Sample ==================")


if __name__ == "__main__":
    main()
