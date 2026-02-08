#!/bin/bash
set -e

# [NOTE] change here for correct storage_cli path
# if storage_base_dir is not set, use default value
if [ -z "${storage_base_dir}" ]; then
  storage_base_dir=/home/vecglypher/mnt/
fi
tokenizer_path=workspace/hf_downloads/Qwen/Qwen3-4B


python src/svg_glyph_gen_v2/sample_index.py \
    data/processed/split_train_test_index/alphanumeric/font_family/train.tsv \
    100 \
    data/processed/split_train_test_index/alphanumeric/font_family/train-sample_100.tsv
python src/svg_glyph_gen_v2/sample_index.py \
    data/processed/split_train_test_index/alphanumeric/font_family/ood_test.tsv \
    30 \
    data/processed/split_train_test_index/alphanumeric/font_family/ood_test-sample_30.tsv

# optimize svg
input_content_file=data/processed/content/oxford-5000.txt
font_split_tsv=data/processed/split_train_test_index/alphanumeric/font_family/train-sample_100.tsv
output_dir=data/processed/normalized_svg-250903-oxford_5000-100_fonts-train-sample_100
python -m src.svg_glyph_gen_v2.batch_render_normalized_svg_v2 \
--input_content_file $input_content_file --font_split_tsv=${font_split_tsv} --output_dir=${output_dir} \
--num_workers=40 --overwrite True --metadata_batch_size 10


input_content_file=data/processed/content/oxford-5000.txt
font_split_tsv=data/processed/split_train_test_index/alphanumeric/font_family/ood_test-sample_30.tsv
output_dir=data/processed/normalized_svg-250903-oxford_5000-100_fonts-ood_test-sample_30
python -m src.svg_glyph_gen_v2.batch_render_normalized_svg_v2 \
--input_content_file $input_content_file --font_split_tsv=${font_split_tsv} --output_dir=${output_dir} \
--num_workers=40 --overwrite True --metadata_batch_size 10

# build sft data
apply_word_sep=True

sft_base_dir=data/processed/sft/250903-oxford_5000-100_fonts-apply_word_sep/

split_name=train-sample_100
font_split_tsv=data/processed/split_train_test_index/alphanumeric/font_family/train-sample_100.tsv
input_svg_dir=data/processed/normalized_svg-250903-oxford_5000-100_fonts-train-sample_100/
input_content_file=data/processed/content/oxford-5000.txt
python -m src.svg_glyph_gen_v2.build_sft_data_v2 \
    --font_split_tsv ${font_split_tsv} \
    --input_svg_dir ${input_svg_dir} \
    --input_content_file ${input_content_file} \
    --apply_word_sep ${apply_word_sep} \
    --output_dir "${sft_base_dir}/${split_name}"

num_contents=600
split_name=ood_test-sample_30-contents_${num_contents}
font_split_tsv=data/processed/split_train_test_index/alphanumeric/font_family/ood_test-sample_30.tsv
input_svg_dir=data/processed/normalized_svg-250903-oxford_5000-100_fonts-ood_test-sample_30/
input_content_file=data/processed/content/oxford-5000.txt
python -m src.svg_glyph_gen_v2.build_sft_data_v2 \
    --font_split_tsv ${font_split_tsv} \
    --input_svg_dir ${input_svg_dir} \
    --input_content_file ${input_content_file} \
    --apply_word_sep ${apply_word_sep} \
    --num_contents ${num_contents} \
    --output_dir "${sft_base_dir}/${split_name}"

split_name=train-alphanumeric
font_split_tsv=data/processed/split_train_test_index/alphanumeric/font_family/train.tsv
input_svg_dir=data/processed/normalized_svg/
input_content_file=data/processed/content/alphanumeric.txt
python -m src.svg_glyph_gen_v2.build_sft_data_v2 \
    --font_split_tsv ${font_split_tsv} \
    --input_svg_dir ${input_svg_dir} \
    --input_content_file ${input_content_file} \
    --apply_word_sep ${apply_word_sep} \
    --output_dir "${sft_base_dir}/${split_name}"

split_name=ood_test-alphanumeric
font_split_tsv=data/processed/split_train_test_index/alphanumeric/font_family/ood_test.tsv
input_svg_dir=data/processed/normalized_svg/
input_content_file=data/processed/content/alphanumeric.txt
python -m src.svg_glyph_gen_v2.build_sft_data_v2 \
    --font_split_tsv ${font_split_tsv} \
    --input_svg_dir ${input_svg_dir} \
    --input_content_file ${input_content_file} \
    --apply_word_sep ${apply_word_sep} \
    --output_dir "${sft_base_dir}/${split_name}"

# Stat token len
split_names=(
train-sample_100
ood_test-sample_30-contents_${num_contents}
train-alphanumeric
ood_test-alphanumeric
)
for split_name in ${split_names[@]}; do
TOKENIZERS_PARALLELISM=true python -m src.svg_glyph_gen_v2.stat_token_len \
--dataset_dir ${sft_base_dir} \
--dataset_name ${split_name} \
--output_dir ${sft_base_dir}
# --model_name_or_path "${storage_base_dir}/${tokenizer_path}"
done

# Validate no overlap font family
python -m src.svg_glyph_gen_v2.validate_no_overlap_font_family \
  -a ${sft_base_dir}/"train-sample_100" \
  -b ${sft_base_dir}/"ood_test-sample_30-contents_${num_contents}" \
  --output_dir ${sft_base_dir}/overlap_font_family
python -m src.svg_glyph_gen_v2.validate_no_overlap_font_family \
  -a ${sft_base_dir}/"train-sample_100" \
  -b ${sft_base_dir}/"ood_test-alphanumeric" \
  --output_dir ${sft_base_dir}/overlap_font_family
python -m src.svg_glyph_gen_v2.validate_no_overlap_font_family \
  -a ${sft_base_dir}/"train-alphanumeric" \
  -b ${sft_base_dir}/"ood_test-sample_30-contents_${num_contents}" \
  --output_dir ${sft_base_dir}/overlap_font_family
python -m src.svg_glyph_gen_v2.validate_no_overlap_font_family \
  -a ${sft_base_dir}/"train-alphanumeric" \
  -b ${sft_base_dir}/"ood_test-alphanumeric" \
  --output_dir ${sft_base_dir}/overlap_font_family


# create dataset_info.json for llama-factory
python src/svg_glyph_gen_v2/build_dataset_info.py ${sft_base_dir}
python src/svg_glyph_gen_v2/stat_sft_data.py ${sft_base_dir}


# filter by token len
filtered_sft_base_dir=data/processed/filtered_sft/250903-oxford_5000-100_fonts-apply_word_sep/

# filter for oxford.
max_token_len=6000 # NOTE: check it accroding to .../sft/*/stat_token_len/*train*.pdf, @ 99 %ile
num_contents=600
split_names=(
train-sample_100
ood_test-sample_30-contents_${num_contents}
)
for split_name in "${split_names[@]}"; do
TOKENIZERS_PARALLELISM=true python src/svg_glyph_gen_v2/filter_by_token_len.py \
  "${sft_base_dir}/${split_name}" \
  "${filtered_sft_base_dir}/${split_name}" \
  "${storage_base_dir}/${tokenizer_path}" \
  --max_token_len ${max_token_len} \
  --num_worker 20 \
  --add_special_tokens "<|SEP|>"
done

# filter for alaphanumeric. different max_token_len
max_token_len=1000 # NOTE: check it accroding to .../sft/*/stat_token_len/*train*.pdf, @ 99 %ile
split_names=(
train-alphanumeric
ood_test-alphanumeric
)
for split_name in "${split_names[@]}"; do
python src/svg_glyph_gen_v2/filter_by_token_len.py \
  "${sft_base_dir}/${split_name}" \
  "${filtered_sft_base_dir}/${split_name}" \
  "${storage_base_dir}/${tokenizer_path}" \
  --max_token_len ${max_token_len} \
  --num_worker 20 \
  --add_special_tokens "<|SEP|>"
done

# create dataset_info.json for llama-factory
python src/svg_glyph_gen_v2/build_dataset_info.py ${filtered_sft_base_dir}
python src/svg_glyph_gen_v2/stat_sft_data.py ${filtered_sft_base_dir}


# upload to storage_cli
set +e  # Temporarily disable exit on error for storage_cli commands

storage_cli mkdirs workspace/svg_glyph_llm/data/processed/sft/ || true
storage_cli  --prod-use-cython-client putr data/processed/sft/ workspace/svg_glyph_llm/data/processed/sft/ -j 10 --threads 20 || true

storage_cli mkdirs workspace/svg_glyph_llm/data/processed/filtered_sft/ || true
storage_cli  --prod-use-cython-client putr data/processed/filtered_sft/ workspace/svg_glyph_llm/data/processed/filtered_sft/ -j 10 --threads 20 || true

set -e
