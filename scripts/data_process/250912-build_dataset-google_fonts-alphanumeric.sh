#!/bin/bash
set -e

# [NOTE] change here for correct storage_cli path
# if storage_base_dir is not set, use default value
if [ -z "${storage_base_dir}" ]; then
  storage_base_dir=/home/vecglypher/mnt/
fi
tokenizer_path=workspace/hf_downloads/Qwen/Qwen3-4B


# prepare content
python -m src.svg_glyph_gen_v2.gather_content \
  data/processed/content

# prepare font split
python -m src.svg_glyph_gen_v2.split_train_test_index_v2 \
  --input_type font_family \
  --num_ind_test 120 \
  --num_ood_test 120 \
  --gfont_metadata_path data/processed/google_font_metadata-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr \
  --content_path data/processed/content/alphanumeric.txt \
  --output_dir data/processed/split_train_test_index/alphanumeric
python -m src.svg_glyph_gen_v2.split_train_test_index_v2 \
  --input_type content \
  --num_ind_test 10 \
  --num_ood_test 0 \
  --gfont_metadata_path data/processed/google_font_metadata-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr \
  --content_path data/processed/content/alphanumeric.txt \
  --output_dir data/processed/split_train_test_index/alphanumeric

# render svg
input_content_file=data/processed/content/alphanumeric.txt
input_metadata_jsonl=data/processed/google_font_metadata-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr
output_dir=data/processed/normalized_svg
python -m src.svg_glyph_gen_v2.batch_render_normalized_svg_v2 \
--input_metadata_jsonl $input_metadata_jsonl \
--input_content_file $input_content_file \
--input_google_font_dir data/google_fonts/ofl \
--output_dir ${output_dir} \
--num_workers=20

# build sft data
sft_base_dir=data/processed/sft/250903-alphanumeric/

input_svg_dir=data/processed/normalized_svg
split_names=(
  "train_font_family"
  "ood_font_family"
)
font_split_tsvs=(
  "data/processed/split_train_test_index/alphanumeric/font_family/train.tsv"
  "data/processed/split_train_test_index/alphanumeric/font_family/ood_test.tsv"
)
for i in "${!split_names[@]}"; do
  split_name="${split_names[$i]}"
  font_split_tsv="${font_split_tsvs[$i]}"
  python -m src.svg_glyph_gen_v2.build_sft_data_v2 \
    --font_split_tsv "${font_split_tsv}" \
    --output_dir "${sft_base_dir}/${split_name}" \
    --input_metadata_jsonl "${input_metadata_jsonl}" \
    --input_content_file "${input_content_file}" \
    --input_svg_dir ${input_svg_dir} \
    --output_log_dir "${sft_base_dir}" \
    --num_workers 20
done

# Stat token len
split_names=(
  "train_font_family"
  "ood_font_family"
)
for split_name in ${split_names[@]}; do
TOKENIZERS_PARALLELISM=true python -m src.svg_glyph_gen_v2.stat_token_len \
--dataset_dir ${sft_base_dir} \
--dataset_name ${split_name} \
--output_dir ${sft_base_dir}/stat_token_len/${split_name}
# --model_name_or_path "${storage_base_dir}/${tokenizer_path}"
done

# Validate no overlap font family
python -m src.svg_glyph_gen_v2.validate_no_overlap_font_family \
  -a ${sft_base_dir}/"train_font_family" \
  -b ${sft_base_dir}/"ood_font_family" \
  --output_dir ${sft_base_dir}/overlap_font_family

# check baseline across different fonts
for target_content_str in "a" "g" "f"; do
python -m src.svg_glyph_gen_v2.display_rendered_svg_in_line \
--input_svg_dir ${sft_base_dir}/ood_font_family/ \
--output_dir ${sft_base_dir}/svg_in_line/${target_content_str} \
--target_content_str ${target_content_str}
done

# create dataset_info.json for llama-factory
python src/svg_glyph_gen_v2/build_dataset_info.py ${sft_base_dir}
python src/svg_glyph_gen_v2/stat_sft_data.py ${sft_base_dir}


# filter by token len
filtered_sft_base_dir=data/processed/filtered_sft/250903-alphanumeric/

# filter for alphanumeric.
max_token_len=1300 # NOTE: check it accroding to .../sft/*/stat_token_len/*train*.pdf, @ 99 %ile
split_name=train_font_family
TOKENIZERS_PARALLELISM=true python src/svg_glyph_gen_v2/filter_by_token_len.py \
  "${sft_base_dir}/${split_name}" \
  "${filtered_sft_base_dir}/${split_name}" \
  --max_token_len ${max_token_len} \
  --num_worker 20 \
  --add_special_tokens "<|SEP|>"
# "${storage_base_dir}/${tokenizer_path}" \
# copy ood_font_family
cp -r "${sft_base_dir}/ood_font_family" "${filtered_sft_base_dir}/ood_font_family"

# create dataset_info.json for llama-factory
python src/svg_glyph_gen_v2/build_dataset_info.py ${filtered_sft_base_dir}
python src/svg_glyph_gen_v2/stat_sft_data.py ${filtered_sft_base_dir}



# TODO: upload to storage_cli, use another script
# # upload to storage_cli
# set +e  # Temporarily disable exit on error for storage_cli commands

# storage_cli mkdirs workspace/svg_glyph_llm/data/processed/sft/ || true
# storage_cli  --prod-use-cython-client putr data/processed/sft/ workspace/svg_glyph_llm/data/processed/sft/ -j 10 --threads 20 || true


# storage_cli mkdirs workspace/svg_glyph_llm/data/processed/filtered_sft/ || true
# storage_cli  --prod-use-cython-client putr data/processed/filtered_sft/ workspace/svg_glyph_llm/data/processed/filtered_sft/ -j 10 --threads 20 || true

# set -e  # Re-enable exit on error
