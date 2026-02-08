#!/bin/bash
set -e


# =======================================
input_dir=data/processed/filtered_sft/250903-alphanumeric/train_font_family
decoded_input_dir=misc/250903-alphanumeric-ref_img/tmp/train_font_family-decoded
img_base64_dir=misc/250903-alphanumeric-ref_img/tmp/train_font_family-decoded-img_base64
output_dir=misc/250903-alphanumeric-ref_img/
output_dataset_name=train_font_family

python -m src.serve.decode_to_svg "${input_dir}" "${decoded_input_dir}"
python -m src.eval.svg2img_dir "${decoded_input_dir}" "${img_base64_dir}" --field output --width 192 --height 192
python -m src.svg_glyph_gen_v2.build_ref_img_sft_data \
    --input_infer_jsonl_dir "${input_dir}" \
    --input_decoded_infer_jsonl_dir "${decoded_input_dir}" \
    --input_infer_img_base64_dir "${img_base64_dir}" \
    --output_dir "${output_dir}" \
    --output_dataset_name "${output_dataset_name}"


input_dir=data/processed/filtered_sft/250903-alphanumeric/ood_font_family_decon
decoded_input_dir=misc/250903-alphanumeric-ref_img/tmp/ood_font_family_decon-decoded
img_base64_dir=misc/250903-alphanumeric-ref_img/tmp/ood_font_family_decon-decoded-img_base64
output_dir=misc/250903-alphanumeric-ref_img/
output_dataset_name=ood_font_family_decon

python -m src.serve.decode_to_svg "${input_dir}" "${decoded_input_dir}"
python -m src.eval.svg2img_dir "${decoded_input_dir}" "${img_base64_dir}" --field output --width 192 --height 192
python -m src.svg_glyph_gen_v2.build_ref_img_sft_data \
    --input_infer_jsonl_dir "${input_dir}" \
    --input_decoded_infer_jsonl_dir "${decoded_input_dir}" \
    --input_infer_img_base64_dir "${img_base64_dir}" \
    --output_dir "${output_dir}" \
    --output_dataset_name "${output_dataset_name}"

# infer data
input_dir=data/processed/filtered_sft/250903-alphanumeric/ood_font_family_decon
decoded_input_dir=misc/250903-alphanumeric-ref_img/tmp/ood_font_family_decon-decoded
img_base64_dir=misc/250903-alphanumeric-ref_img/tmp/ood_font_family_decon-decoded-img_base64
output_dir=misc/250903-alphanumeric-ref_img-b64_pil/
output_dataset_name=ood_font_family_decon
python -m src.svg_glyph_gen_v2.build_ref_img_sft_data_b64_pil \
    --input_infer_jsonl_dir "${input_dir}" \
    --input_decoded_infer_jsonl_dir "${decoded_input_dir}" \
    --input_infer_img_base64_dir "${img_base64_dir}" \
    --output_dir "${output_dir}" \
    --output_dataset_name "${output_dataset_name}"


# =======================================
input_dir=data/processed/filtered_sft/250910-alphanumeric-abs_coord/train_font_family
decoded_input_dir=misc/250910-alphanumeric-abs_coord-ref_img/tmp/train_font_family-decoded
img_base64_dir=misc/250910-alphanumeric-abs_coord-ref_img/tmp/train_font_family-decoded-img_base64
output_dir=misc/250910-alphanumeric-abs_coord-ref_img/
output_dataset_name=train_font_family

python -m src.serve.decode_to_svg "${input_dir}" "${decoded_input_dir}"
python -m src.eval.svg2img_dir "${decoded_input_dir}" "${img_base64_dir}" --field output --width 192 --height 192
python -m src.svg_glyph_gen_v2.build_ref_img_sft_data \
    --input_infer_jsonl_dir "${input_dir}" \
    --input_decoded_infer_jsonl_dir "${decoded_input_dir}" \
    --input_infer_img_base64_dir "${img_base64_dir}" \
    --output_dir "${output_dir}" \
    --output_dataset_name "${output_dataset_name}"


input_dir=data/processed/filtered_sft/250910-alphanumeric-abs_coord/ood_font_family_decon
decoded_input_dir=misc/250910-alphanumeric-abs_coord-ref_img/tmp/ood_font_family_decon-decoded
img_base64_dir=misc/250910-alphanumeric-abs_coord-ref_img/tmp/ood_font_family_decon-decoded-img_base64
output_dir=misc/250910-alphanumeric-abs_coord-ref_img/
output_dataset_name=ood_font_family_decon

python -m src.serve.decode_to_svg "${input_dir}" "${decoded_input_dir}"
python -m src.eval.svg2img_dir "${decoded_input_dir}" "${img_base64_dir}" --field output --width 192 --height 192
python -m src.svg_glyph_gen_v2.build_ref_img_sft_data \
    --input_infer_jsonl_dir "${input_dir}" \
    --input_decoded_infer_jsonl_dir "${decoded_input_dir}" \
    --input_infer_img_base64_dir "${img_base64_dir}" \
    --output_dir "${output_dir}" \
    --output_dataset_name "${output_dataset_name}"

# infer data
input_dir=data/processed/filtered_sft/250910-alphanumeric-abs_coord/ood_font_family_decon
decoded_input_dir=misc/250910-alphanumeric-abs_coord-ref_img/tmp/ood_font_family_decon-decoded
img_base64_dir=misc/250910-alphanumeric-abs_coord-ref_img/tmp/ood_font_family_decon-decoded-img_base64
output_dir=misc/250910-alphanumeric-abs_coord-ref_img-b64_pil/
output_dataset_name=ood_font_family_decon
python -m src.svg_glyph_gen_v2.build_ref_img_sft_data_b64_pil \
    --input_infer_jsonl_dir "${input_dir}" \
    --input_decoded_infer_jsonl_dir "${decoded_input_dir}" \
    --input_infer_img_base64_dir "${img_base64_dir}" \
    --output_dir "${output_dir}" \
    --output_dataset_name "${output_dataset_name}"


# =======================================
wait
output_dir=misc/250903-alphanumeric-ref_img/
python src/svg_glyph_gen_v2/build_dataset_info.py --add_images "${output_dir}"
python src/svg_glyph_gen_v2/stat_sft_data.py "${output_dir}"

output_dir=misc/250910-alphanumeric-abs_coord-ref_img/
python src/svg_glyph_gen_v2/build_dataset_info.py --add_images "${output_dir}"
python src/svg_glyph_gen_v2/stat_sft_data.py "${output_dir}"

output_dir=misc/250903-alphanumeric-ref_img-b64_pil/
python src/svg_glyph_gen_v2/build_dataset_info.py --add_images "${output_dir}"
python src/svg_glyph_gen_v2/stat_sft_data.py "${output_dir}"

output_dir=misc/250910-alphanumeric-abs_coord-ref_img-b64_pil/
python src/svg_glyph_gen_v2/build_dataset_info.py --add_images "${output_dir}"
python src/svg_glyph_gen_v2/stat_sft_data.py "${output_dir}"


# =======================================
exit


storage_cli --prod-use-cython-client putr --threads 128 --jobs 100 misc/250903-alphanumeric-ref_img workspace/svg_glyph_llm/data/250903-alphanumeric-ref_img
storage_cli --prod-use-cython-client putr --threads 128 --jobs 100 misc/250910-alphanumeric-abs_coord-ref_img workspace/svg_glyph_llm/data/250910-alphanumeric-abs_coord-ref_img

storage_cli --prod-use-cython-client putr --threads 128 --jobs 100 misc/250903-alphanumeric-ref_img-b64_pil workspace/svg_glyph_llm/data/250903-alphanumeric-ref_img-b64_pil
storage_cli --prod-use-cython-client putr --threads 128 --jobs 100 misc/250910-alphanumeric-abs_coord-ref_img-b64_pil workspace/svg_glyph_llm/data/250910-alphanumeric-abs_coord-ref_img-b64_pil

# pretokenzation does not include images, we need to convert image path to PIL
# python scripts/sft_on_cluster_submitter/submit.py dry_run=true local_run=true
# cp misc/submitter_artifacts/*-default_job/launch_cmd.sh misc/tokenize_dataset.sh

# dataset=train_font_family \
# eval_dataset=ood_font_family_decon \
# max_steps=10 \
# model_name_or_path=/home/vecglypher/mnt/workspace/hf_downloads/google/gemma-3-4b-it \
# template=gemma3 \

# dataset_dir=misc/250903-alphanumeric-ref_img \
# tokenized_path=misc/250903-alphanumeric-ref_img-gemma3-tokenized \

# dataset_dir=misc/250910-alphanumeric-abs_coord-ref_img \
# tokenized_path=misc/250910-alphanumeric-abs_coord-ref_img-gemma3-tokenized \

storage_cli --prod-use-cython-client putr --threads 20 --jobs 10 misc/250903-alphanumeric-ref_img-gemma3-tokenized workspace/svg_glyph_llm/data/250903-alphanumeric-ref_img-gemma3-tokenized
storage_cli --prod-use-cython-client putr --threads 20 --jobs 10 misc/250910-alphanumeric-abs_coord-ref_img-gemma3-tokenized workspace/svg_glyph_llm/data/250910-alphanumeric-abs_coord-ref_img-gemma3-tokenized


python src/svg_glyph_gen_v2/load_img_for_hf_dataset.py misc/250903-alphanumeric-ref_img-gemma3-tokenized
python src/svg_glyph_gen_v2/load_img_for_hf_dataset.py misc/250910-alphanumeric-abs_coord-ref_img-gemma3-tokenized

storage_cli --prod-use-cython-client putr --threads 20 --jobs 10 misc/250903-alphanumeric-ref_img-gemma3-tokenized-pil workspace/svg_glyph_llm/data/250903-alphanumeric-ref_img-gemma3-tokenized-pil
storage_cli --prod-use-cython-client putr --threads 20 --jobs 10 misc/250910-alphanumeric-abs_coord-ref_img-gemma3-tokenized-pil workspace/svg_glyph_llm/data/250910-alphanumeric-abs_coord-ref_img-gemma3-tokenized-pil
