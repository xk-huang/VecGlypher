#!/bin/bash
set -e


# google fonts train vs google fonts test
python -m src.tools.check_data_contamination \
    -a data/processed_envato/filtered_sft/250903-alphanumeric/train_font_family \
    -b data/processed_envato/filtered_sft/250903-alphanumeric/ood_font_family

# google fonts train vs google fonts test
python -m src.tools.check_data_contamination \
    -a data/processed_envato/filtered_sft/250910-alphanumeric-abs_coord/train_font_family \
    -b data/processed_envato/filtered_sft/250910-alphanumeric-abs_coord/ood_font_family

# envato fonts train vs google fonts test
python -m src.tools.check_data_contamination \
    -a data/processed_envato/filtered_sft/250903-envato-alphanumeric/train_font_family \
    -b data/processed_envato/filtered_sft/250903-alphanumeric/ood_font_family

# envato fonts train vs google fonts test
python -m src.tools.check_data_contamination \
    -a data/processed_envato/filtered_sft/250903-envato-alphanumeric-abs_coord/train_font_family \
    -b data/processed_envato/filtered_sft/250910-alphanumeric-abs_coord/ood_font_family

# envato fonts train vs envato fonts test
python -m src.tools.check_data_contamination \
    -a data/processed_envato/filtered_sft/250903-envato-alphanumeric/train_font_family \
    -b data/processed_envato/filtered_sft/250903-envato-alphanumeric/ood_font_family

# envato fonts train vs envato fonts test
python -m src.tools.check_data_contamination \
    -a data/processed_envato/filtered_sft/250903-envato-alphanumeric-abs_coord/train_font_family \
    -b data/processed_envato/filtered_sft/250903-envato-alphanumeric-abs_coord/ood_font_family


# decontaminate google fonts
dataset_dir=data/processed/filtered_sft/250903-alphanumeric
python -m src.tools.decontaminate_data \
    -a "${dataset_dir}/train_font_family" \
    -b "${dataset_dir}/ood_font_family"

python src/svg_glyph_gen_v2/build_dataset_info.py "${dataset_dir}"
python src/svg_glyph_gen_v2/stat_sft_data.py "${dataset_dir}"


dataset_dir=data/processed/filtered_sft/250910-alphanumeric-abs_coord
python -m src.tools.decontaminate_data \
    -a "${dataset_dir}/train_font_family" \
    -b "${dataset_dir}/ood_font_family"

python src/svg_glyph_gen_v2/build_dataset_info.py "${dataset_dir}"
python src/svg_glyph_gen_v2/stat_sft_data.py "${dataset_dir}"


# decontaminate google fonts test in envato google mix
dataset_dir=data/processed_envato/filtered_sft/250903-alphanumeric
python -m src.tools.decontaminate_data \
    -a "${dataset_dir}/train_font_family" \
    -b "${dataset_dir}/ood_font_family"

python src/svg_glyph_gen_v2/build_dataset_info.py "${dataset_dir}"
python src/svg_glyph_gen_v2/stat_sft_data.py "${dataset_dir}"


dataset_dir=data/processed_envato/filtered_sft/250910-alphanumeric-abs_coord
python -m src.tools.decontaminate_data \
    -a "${dataset_dir}/train_font_family" \
    -b "${dataset_dir}/ood_font_family"

python src/svg_glyph_gen_v2/build_dataset_info.py "${dataset_dir}"
python src/svg_glyph_gen_v2/stat_sft_data.py "${dataset_dir}"


python src/svg_glyph_gen_v2/merge_dataest_info_stats.py data/processed_envato/filtered_sft


# double check after data decontamination

# google fonts train vs google fonts test
python -m src.tools.check_data_contamination \
    -a data/processed_envato/filtered_sft/250903-alphanumeric/train_font_family \
    -b data/processed_envato/filtered_sft/250903-alphanumeric/ood_font_family_decon

# google fonts train vs google fonts test
python -m src.tools.check_data_contamination \
    -a data/processed_envato/filtered_sft/250910-alphanumeric-abs_coord/train_font_family \
    -b data/processed_envato/filtered_sft/250910-alphanumeric-abs_coord/ood_font_family_decon

# google fonts train vs google fonts test
python -m src.tools.check_data_contamination \
    -a data/processed/filtered_sft/250903-alphanumeric/train_font_family \
    -b data/processed/filtered_sft/250903-alphanumeric/ood_font_family_decon

# google fonts train vs google fonts test
python -m src.tools.check_data_contamination \
    -a data/processed/filtered_sft/250910-alphanumeric-abs_coord/train_font_family \
    -b data/processed/filtered_sft/250910-alphanumeric-abs_coord/ood_font_family_decon

exit

# upload to google
storage_cli --prod-use-cython-client putr -j 10 --threads 20 data/processed/filtered_sft/250903-alphanumeric/ood_font_family_decon workspace/svg_glyph_llm/data/processed/filtered_sft/250903-alphanumeric/ood_font_family_decon
storage_cli --prod-use-cython-client put --threads 20 data/processed/filtered_sft/250903-alphanumeric/dataset_info.json workspace/svg_glyph_llm/data/processed/filtered_sft/250903-alphanumeric/dataset_info.json --overwrite
storage_cli --prod-use-cython-client put --threads 20 data/processed/filtered_sft/250903-alphanumeric/dataset_stat.json workspace/svg_glyph_llm/data/processed/filtered_sft/250903-alphanumeric/dataset_stat.json --overwrite

storage_cli --prod-use-cython-client putr -j 10 --threads 20 data/processed/filtered_sft/250910-alphanumeric-abs_coord/ood_font_family_decon workspace/svg_glyph_llm/data/processed/filtered_sft/250910-alphanumeric-abs_coord/ood_font_family_decon
storage_cli --prod-use-cython-client put --threads 20 data/processed/filtered_sft/250910-alphanumeric-abs_coord/dataset_info.json workspace/svg_glyph_llm/data/processed/filtered_sft/250910-alphanumeric-abs_coord/dataset_info.json --overwrite
storage_cli --prod-use-cython-client put --threads 20 data/processed/filtered_sft/250910-alphanumeric-abs_coord/dataset_stat.json workspace/svg_glyph_llm/data/processed/filtered_sft/250910-alphanumeric-abs_coord/dataset_stat.json --overwrite

# upload to envato google mix
storage_cli --prod-use-cython-client putr -j 10 --threads 20 data/processed_envato/filtered_sft/250903-alphanumeric workspace/svg_glyph_llm/data/processed_envato/filtered_sft/250903-alphanumeric --overwrite
storage_cli --prod-use-cython-client putr -j 10 --threads 20 data/processed_envato/filtered_sft/250910-alphanumeric-abs_coord workspace/svg_glyph_llm/data/processed_envato/filtered_sft/250910-alphanumeric-abs_coord --overwrite

storage_cli --prod-use-cython-client put --threads 20 data/processed_envato/filtered_sft/dataset_info.json workspace/svg_glyph_llm/data/processed_envato/filtered_sft/dataset_info.json --overwrite
storage_cli --prod-use-cython-client put --threads 20 data/processed_envato/filtered_sft/dataset_stat.json workspace/svg_glyph_llm/data/processed_envato/filtered_sft/dataset_stat.json --overwrite
