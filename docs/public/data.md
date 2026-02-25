# Data

We release the processed datasets for training and evaluation.

Google Fonts (training and evluation): https://huggingface.co/datasets/VecGlypher/Google-Fonts-Dataset

Note on Envato Fonts: For license considerations, Envato Fonts are not included in the release, but the processing scripts are provided for users to build the datasets on their own. You can also use [MyFont](https://www.cs.rochester.edu/u/tchen45/font/font.html) as an alternative source of textual descriptions for font data ([quick link](https://drive.google.com/open?id=10GRqLu6-1JPXI8rcq23S4-4AhB6On-L6), [HF mirror](https://huggingface.co/datasets/VecGlypher/MyFont-Mirror)).


# Data Preparation

Dataset tools live under `src/svg_glyph_gen_v2/`. End-to-end pipelines are in
`scripts/data_process/`. Run `--help` on any tool for required inputs.

## Example workflow

```bash
python -m src.svg_glyph_gen_v2.gather_content data/processed/content
python -m src.svg_glyph_gen_v2.batch_render_normalized_svg_v2 --help
python -m src.svg_glyph_gen_v2.build_sft_data_v2 --help
python src/svg_glyph_gen_v2/build_dataset_info.py data/processed/sft/<dataset_name>
```

## Notes

- `build_dataset_info.py` writes `dataset_info.json`, which is required for
  training configs that point to `dataset_dir`.
- Use `scripts/data_process/*.sh` as reference pipelines and update paths for
  your local setup.

## Example: Google Fonts

Download Google Fonts
```bash
wget https://github.com/google/fonts/archive/44a3c9a8d8a5b3d6adadedcae000e40e520c55d7.zip -O data/google_fonts.zip
unzip -l data/google_fonts.zip

unzip data/google_fonts.zip -d data
mv data/fonts-44a3c9a8d8a5b3d6adadedcae000e40e520c55d7 data/google_fonts
```

Download Qwen-VL model weights for OCR filtering

```bash
local_dir=saves/
mkdir -p $local_dir

models=(
    # "Qwen/Qwen2.5-VL-7B-Instruct"
    # "Qwen/Qwen2.5-VL-32B-Instruct"
    "Qwen/Qwen3-VL-30B-A3B-Instruct"
    # "Qwen/Qwen3-VL-4B-Instruct"
    # "Qwen/Qwen3-VL-8B-Instruct"
    "Qwen/Qwen3-4B"
)
for model in "${models[@]}"; do
    hf download "$model" --repo-type model --local-dir "$local_dir/$model"
done
```

Process Google Fonts (See scripts/setup_env/process_all_data.sh)

```bash
bash scripts/data_process/250912-build_metadata-google_fonts.sh

bash scripts/data_process/250912-build_dataset-google_fonts-alphanumeric.sh
bash scripts/data_process/250912-build_dataset-google_fonts-alphanumeric-abs_coord.sh
bash scripts/data_process/251023-build_dataset-google_fonts-ref_img.sh
bash scripts/data_process/251028-build_dataset-google_fonts-ref_img-eval_dev_split.sh

# Need the tokenized dataset, check the comment int the script for details
# bash scripts/data_process/251026-build_dataset-google_fonts-mix_text_ref_img.sh
```

### Diacritics (Latin)

The diacritics list is defined in `misc/diacritics.py`, and `gather_content`
writes `data/processed/content/diacritics.txt` from it.

```bash
PYTHON=/nfs/vecglypher/miniconda3/envs/svg_glyph_llm_eval/bin/python
bash scripts/data_process/250912-build_dataset-google_fonts-diacritics.sh
```

Outputs are written to `data/processed/normalized_svg-diacritics` and
`data/processed/sft/250912-diacritics`.
