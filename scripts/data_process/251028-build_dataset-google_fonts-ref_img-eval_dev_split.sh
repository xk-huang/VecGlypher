#!/bin/bash
set -e


# =======================================
# storage_cli --prod-use-cython-client getr --threads 20 --jobs 10 workspace/svg_glyph_llm/data/250903-alphanumeric-ref_img-b64_pil/ood_font_family_decon misc/250903-alphanumeric-ref_img-b64_pil/ood_font_family_decon



# =======================================
# storage_cli --prod-use-cython-client getr --threads 128 --jobs 100  workspace/svg_glyph_llm/data/250903-alphanumeric-ref_img misc/250903-alphanumeric-ref_img
# The raw images and svgs of references are in */tmp/*
# we need to add the svg sequence, and the content str of reference images

input_dir=data/processed/filtered_sft/250903-alphanumeric/ood_font_family_decon
# input_dir=/home/vecglypher/mnt/workspace/svg_glyph_llm/data/processed/filtered_sft/250903-alphanumeric/ood_font_family_decon
decoded_input_dir=misc/250903-alphanumeric-ref_img/tmp/ood_font_family_decon-decoded
img_base64_dir=misc/250903-alphanumeric-ref_img/tmp/ood_font_family_decon-decoded-img_base64
output_dir=misc/250903-alphanumeric-ref_img-eval_baselines/
output_dataset_name=ood_font_family_decon

python -m src.svg_glyph_gen_v2.build_ref_img_sft_data \
    --input_infer_jsonl_dir "${input_dir}" \
    --input_decoded_infer_jsonl_dir "${decoded_input_dir}" \
    --input_infer_img_base64_dir "${img_base64_dir}" \
    --output_dir "${output_dir}" \
    --output_dataset_name "${output_dataset_name}"


# infer data
output_dir=misc/250903-alphanumeric-ref_img-eval_baselines-b64_pil/
python -m src.svg_glyph_gen_v2.build_ref_img_sft_data_b64_pil \
    --input_infer_jsonl_dir "${input_dir}" \
    --input_decoded_infer_jsonl_dir "${decoded_input_dir}" \
    --input_infer_img_base64_dir "${img_base64_dir}" \
    --output_dir "${output_dir}" \
    --output_dataset_name "${output_dataset_name}"


# =======================================
python -m src.tools.check_two_jsonl_key \
-a misc/250903-alphanumeric-ref_img/ood_font_family_decon \
-b misc/250903-alphanumeric-ref_img-eval_baselines/ood_font_family_decon \
-k images

python -m src.tools.check_two_jsonl_key \
-a misc/250903-alphanumeric-ref_img-b64_pil/ood_font_family_decon \
-b misc/250903-alphanumeric-ref_img-eval_baselines-b64_pil/ood_font_family_decon \
-k images



# =======================================
mkdir -p  misc/250903-alphanumeric-ref_img-eval_baselines-b64_pil-ood_font_family_decon-dev
ls misc/250903-alphanumeric-ref_img-eval_baselines-b64_pil/ood_font_family_decon


for f in misc/250903-alphanumeric-ref_img-eval_baselines-b64_pil/ood_font_family_decon/chunk_chunk_*.jsonl; do
  base=$(basename "$f")
  head -n 5 "$f" > "misc/250903-alphanumeric-ref_img-eval_baselines-b64_pil-ood_font_family_decon-dev/$base"
done
