#!/bin/bash
set -e

with-proxy python -m src.svg_glyph_gen_v2.gather_content data/processed_envato/content

python -m src.svg_glyph_gen_v2.split_train_test_index_v2 \
  --input_type font_family \
  --num_ind_test 150 \
  --num_ood_test 1000 \
  --gfont_metadata_path data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr \
  --content_path data/processed_envato/content/alphanumeric.txt \
  --output_dir data/processed_envato/split_train_test_index/alphanumeric

python -m src.svg_glyph_gen_v2.split_train_test_index_v2 \
  --input_type content \
  --num_ind_test 10 \
  --num_ood_test 0 \
  --gfont_metadata_path data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr \
  --content_path data/processed_envato/content/alphanumeric.txt \
  --output_dir data/processed_envato/split_train_test_index/alphanumeric


input_content_file=data/processed_envato/content/alphanumeric.txt
input_metadata_jsonl=data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr
input_google_font_dir=../envato_fonts/fonts
output_dir=data/processed_envato/normalized_svg-abs_coord/
python -m src.svg_glyph_gen_v2.batch_render_normalized_svg_v2 \
--input_content_file $input_content_file \
--input_metadata_jsonl $input_metadata_jsonl \
--input_google_font_dir $input_google_font_dir \
--output_dir $output_dir \
--use_relative_path False \
--num_workers=40


# build sft data
sft_base_dir=data/processed_envato/sft/250903-envato-alphanumeric-abs_coord/

split_name="train_font_family"
font_split_tsv=data/processed_envato/split_train_test_index/alphanumeric/font_family/train.tsv
input_svg_dir=data/processed_envato/normalized_svg-abs_coord/
input_content_file=data/processed_envato/content/alphanumeric.txt
input_metadata_jsonl=data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr
python -m src.svg_glyph_gen_v2.build_sft_data_v2 \
    --font_split_tsv ${font_split_tsv} \
    --input_svg_dir ${input_svg_dir} \
    --input_content_file ${input_content_file} \
    --input_metadata_jsonl ${input_metadata_jsonl} \
    --output_dir "${sft_base_dir}/${split_name}" \
    --output_log_dir "${sft_base_dir}" \
    --num_workers 40

split_name="ood_font_family"
font_split_tsv=data/processed_envato/split_train_test_index/alphanumeric/font_family/ood_test.tsv
python -m src.svg_glyph_gen_v2.build_sft_data_v2 \
    --font_split_tsv ${font_split_tsv} \
    --input_svg_dir ${input_svg_dir} \
    --input_content_file ${input_content_file} \
    --input_metadata_jsonl ${input_metadata_jsonl} \
    --output_dir "${sft_base_dir}/${split_name}" \
    --output_log_dir "${sft_base_dir}" \
    --num_workers 40

# Stat token len
split_names=(
train_font_family
ood_font_family
)
for split_name in ${split_names[@]}; do
  if [[ $split_name == "train_font_family" ]]; then
    skip_scatter_flag="--skip_scatter"
  else
    skip_scatter_flag="--no_skip_scatter"
  fi
TOKENIZERS_PARALLELISM=true python -m src.svg_glyph_gen_v2.stat_token_len \
  --dataset_dir ${sft_base_dir} \
  --dataset_name ${split_name} \
  --output_dir ${sft_base_dir}/stat_token_len/${split_name} \
  ${skip_scatter_flag}
# --model_name_or_path "${storage_base_dir}/${tokenizer_path}"
done

# Validate no overlap font family
python -m src.svg_glyph_gen_v2.validate_no_overlap_font_family \
  -a ${sft_base_dir}/"train_font_family" \
  -b ${sft_base_dir}/"ood_font_family" \
  --output_dir ${sft_base_dir}/overlap_font_family

# check baseline across different fonts
for target_content_str in "a" "g" "k"; do
python -m src.svg_glyph_gen_v2.display_rendered_svg_in_line \
--input_svg_dir ${sft_base_dir}/ood_font_family/ \
--output_dir ${sft_base_dir}/svg_in_line/${target_content_str} \
--target_content_str ${target_content_str}
done


# create dataset_info.json for llama-factory
python src/svg_glyph_gen_v2/build_dataset_info.py ${sft_base_dir}
python src/svg_glyph_gen_v2/stat_sft_data.py ${sft_base_dir}


# filter by token len
filtered_sft_base_dir=data/processed_envato/filtered_sft/250903-envato-alphanumeric-abs_coord/

# filter for alphanumeric.
max_token_len=3600 # NOTE: check it accroding to .../sft/*/stat_token_len/*train*.pdf, @ 99 %ile
split_name="train_font_family"
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

# storage_cli mkdirs workspace/svg_glyph_llm/data/processed_envato/sft/ || true
# storage_cli  --prod-use-cython-client putr data/processed_envato/sft/ workspace/svg_glyph_llm/data/processed_envato/sft/ -j 10 --threads 20 || true


# storage_cli mkdirs workspace/svg_glyph_llm/data/processed_envato/filtered_sft/ || true
# storage_cli  --prod-use-cython-client putr data/processed_envato/filtered_sft/ workspace/svg_glyph_llm/data/processed_envato/filtered_sft/ -j 10 --threads 20 || true

# set -e
