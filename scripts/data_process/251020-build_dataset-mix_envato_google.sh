#!/bin/bash
set -e


rsync -avP data/processed/filtered_sft/250903-alphanumeric data/processed_envato/filtered_sft
rsync -avP data/processed/filtered_sft/250910-alphanumeric-abs_coord data/processed_envato/filtered_sft

# repeat dataset for 5 times
datast_dir=data/processed_envato/filtered_sft/250903-alphanumeric
python -m src.svg_glyph_gen_v2.repeat_dataset \
    "${datast_dir}"/train_font_family \
    "${datast_dir}"/train_font_family-5x \
    --num_repeats 5

python src/svg_glyph_gen_v2/build_dataset_info.py "${datast_dir}"
python src/svg_glyph_gen_v2/stat_sft_data.py "${datast_dir}"

datast_dir=data/processed_envato/filtered_sft/250910-alphanumeric-abs_coord
python -m src.svg_glyph_gen_v2.repeat_dataset \
    "${datast_dir}"/train_font_family \
    "${datast_dir}"/train_font_family-5x \
    --num_repeats 5

python src/svg_glyph_gen_v2/build_dataset_info.py "${datast_dir}"
python src/svg_glyph_gen_v2/stat_sft_data.py "${datast_dir}"



python src/svg_glyph_gen_v2/merge_dataest_info_stats.py data/processed_envato/filtered_sft


exit

storage_cli --prod-use-cython-client putr -j 10 --threads 20 data/processed_envato/filtered_sft/250903-alphanumeric workspace/svg_glyph_llm/data/processed_envato/filtered_sft/250903-alphanumeric
storage_cli --prod-use-cython-client putr -j 10 --threads 20 data/processed_envato/filtered_sft/250910-alphanumeric-abs_coord workspace/svg_glyph_llm/data/processed_envato/filtered_sft/250910-alphanumeric-abs_coord

storage_cli --prod-use-cython-client put --threads 20 data/processed_envato/filtered_sft/dataset_info.json workspace/svg_glyph_llm/data/processed_envato/filtered_sft/dataset_info.json
storage_cli --prod-use-cython-client put --threads 20 data/processed_envato/filtered_sft/dataset_stat.json workspace/svg_glyph_llm/data/processed_envato/filtered_sft/dataset_stat.json
