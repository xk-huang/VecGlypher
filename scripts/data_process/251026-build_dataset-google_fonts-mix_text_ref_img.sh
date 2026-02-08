#!/bin/bash
set -e

# storage_cli --prod-use-cython-client getr --threads 20 --jobs 10  workspace/svg_glyph_llm/data/250903-alphanumeric-ref_img-gemma3-tokenized-pil misc/250903-alphanumeric-ref_img-gemma3-tokenized-pil
# storage_cli --prod-use-cython-client getr --threads 20 --jobs 10 workspace/svg_glyph_llm/data/250910-alphanumeric-abs_coord-ref_img-gemma3-tokenized-pil misc/250910-alphanumeric-abs_coord-ref_img-gemma3-tokenized-pil
# storage_cli --prod-use-cython-client getr --threads 20 --jobs 10 workspace/svg_glyph_llm/data/250910-alphanumeric-abs_coord-ref_img-gemma3-tokenized-pil misc/250910-alphanumeric-abs_coord-ref_img-gemma3-tokenized-pil



# pretokenze text dataset
# python scripts/sft_on_cluster_submitter/submit.py dry_run=true local_run=true
# cp misc/submitter_artifacts/*-default_job/launch_cmd.sh misc/tokenize_dataset.sh

# dataset=train_font_family \
# eval_dataset=ood_font_family_decon \
# max_steps=10 \
# model_name_or_path=/home/vecglypher/mnt/workspace/hf_downloads/google/gemma-3-4b-it \
# template=gemma3 \

# output_dir=outputs/debug/tokenize_dataset/250903-alphanumeric-gemma3-tokenized \
# dataset_dir=/home/vecglypher/mnt/workspace/svg_glyph_llm/data/processed/filtered_sft/250903-alphanumeric \
# tokenized_path=misc/250903-alphanumeric-gemma3-tokenized \


# output_dir=outputs/debug/tokenize_dataset/250910-alphanumeric-abs_coord-gemma3-tokenized \
# dataset_dir=/home/vecglypher/mnt/workspace/svg_glyph_llm/data/processed/filtered_sft/250910-alphanumeric-abs_coord \
# tokenized_path=misc/250910-alphanumeric-abs_coord-gemma3-tokenized \

python src/tools/merge_hf_dataset.py \
    -i misc/250903-alphanumeric-gemma3-tokenized \
    -i misc/250903-alphanumeric-ref_img-gemma3-tokenized-pil \
    -o misc/250903-alphanumeric-text_img_merged-gemma3-tokenized-pil

python src/tools/merge_hf_dataset.py \
    -i misc/250910-alphanumeric-abs_coord-gemma3-tokenized \
    -i misc/250910-alphanumeric-abs_coord-ref_img-gemma3-tokenized-pil \
    -o misc/250910-alphanumeric-abs_coord-text_img_merged-gemma3-tokenized-pil


storage_cli --prod-use-cython-client putr --threads 20 --jobs 10 misc/250903-alphanumeric-text_img_merged-gemma3-tokenized-pil workspace/svg_glyph_llm/data/250903-alphanumeric-text_img_merged-gemma3-tokenized-pil
storage_cli --prod-use-cython-client putr --threads 20 --jobs 10 misc/250910-alphanumeric-abs_coord-text_img_merged-gemma3-tokenized-pil workspace/svg_glyph_llm/data/250910-alphanumeric-abs_coord-text_img_merged-gemma3-tokenized-pil
