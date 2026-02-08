#!/bin/bash
set -e

PYTHON=python

$PYTHON -m src.svg_glyph_gen_v2.gather_content \
  data/processed/content

$PYTHON -m src.svg_glyph_gen_v2.split_train_test_index_v2 \
  --input_type font_family \
  --num_ind_test 120 \
  --num_ood_test 120 \
  --gfont_metadata_path data/processed/google_font_metadata-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr \
  --content_path data/processed/content/diacritics.txt \
  --output_dir data/processed/split_train_test_index/diacritics
$PYTHON -m src.svg_glyph_gen_v2.split_train_test_index_v2 \
  --input_type content \
  --num_ind_test 10 \
  --num_ood_test 0 \
  --gfont_metadata_path data/processed/google_font_metadata-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr \
  --content_path data/processed/content/diacritics.txt \
  --output_dir data/processed/split_train_test_index/diacritics

input_content_file=data/processed/content/diacritics.txt
input_metadata_jsonl=data/processed/google_font_metadata-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr
output_dir=data/processed/normalized_svg-diacritics
$PYTHON -m src.svg_glyph_gen_v2.batch_render_normalized_svg_v2 \
  --input_metadata_jsonl $input_metadata_jsonl \
  --input_content_file $input_content_file \
  --input_google_font_dir data/google_fonts/ofl \
  --output_dir ${output_dir} \
  --num_workers=20

sft_base_dir=data/processed/sft/250912-diacritics/

input_svg_dir=data/processed/normalized_svg-diacritics
split_names=(
  "train_font_family"
  "ood_font_family"
)
font_split_tsvs=(
  "data/processed/split_train_test_index/diacritics/font_family/train.tsv"
  "data/processed/split_train_test_index/diacritics/font_family/ood_test.tsv"
)
for i in "${!split_names[@]}"; do
  split_name="${split_names[$i]}"
  font_split_tsv="${font_split_tsvs[$i]}"
  $PYTHON -m src.svg_glyph_gen_v2.build_sft_data_v2 \
    --font_split_tsv "${font_split_tsv}" \
    --output_dir "${sft_base_dir}/${split_name}" \
    --input_metadata_jsonl "${input_metadata_jsonl}" \
    --input_content_file "${input_content_file}" \
    --input_svg_dir ${input_svg_dir} \
    --output_log_dir "${sft_base_dir}" \
    --num_workers 20
done

$PYTHON -m src.svg_glyph_gen_v2.validate_no_overlap_font_family \
  -a ${sft_base_dir}/"train_font_family" \
  -b ${sft_base_dir}/"ood_font_family" \
  --output_dir ${sft_base_dir}/overlap_font_family

$PYTHON src/svg_glyph_gen_v2/build_dataset_info.py ${sft_base_dir}
$PYTHON src/svg_glyph_gen_v2/stat_sft_data.py ${sft_base_dir}
