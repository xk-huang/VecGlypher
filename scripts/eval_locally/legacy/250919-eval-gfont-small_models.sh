#!/bin/bash
set -e

output_base_dir=outputs/250919-eval-gfont-small_models
storage_base_dir=workspace/svg_glyph_llm/saves/legacy

model_base_dir=saves
data_dir=/home/vecglypher/mnt/workspace/svg_glyph_llm/data/legacy/251009-processed/filtered_sft/250903-alphanumeric/ood_font_family

models=(
#
250910-google_font-ablate_svg_repr/Qwen3-1_7B-abs_coord
250910-google_font-ablate_svg_repr/Qwen3-1_7B-rel_coord
250910-google_font-ablate_svg_repr/Qwen3-4B-abs_coord
250910-google_font-ablate_svg_repr/Qwen3-4B-rel_coord
250910-google_font-ablate_svg_repr/Qwen3-8B-abs_coord
250910-google_font-ablate_svg_repr/Qwen3-8B-rel_coord
#
250914-envato-ablate_scale/Qwen3-1_7B
250914-envato-ablate_scale/Qwen3-4B
250914-envato-ablate_scale/Qwen3-8B
#
250915-envato-ablate_hparam/Qwen3-1_7B-bs_256
250915-envato-ablate_hparam/Qwen3-1_7B-lr_decay_0_1
)

for model in "${models[@]}"; do
bash scripts/eval_locally/eval_suite-template.sh \
    --storage_model_path="${storage_base_dir}/${model}" \
    --model_path="${model_base_dir}/${model}" \
    --output_dir="${output_base_dir}/${model}/t_0_0" \
    --data="${data_dir}" \
    --temperature=0.0
done

for model in "${models[@]}"; do
bash scripts/eval_locally/eval_suite-template.sh \
    --storage_model_path="${storage_base_dir}/${model}" \
    --model_path="${model_base_dir}/${model}" \
    --output_dir="${output_base_dir}/${model}/t_0_7" \
    --data="${data_dir}" \
    --temperature=0.7
done

for model in "${models[@]}"; do
bash scripts/eval_locally/eval_suite-template.sh \
    --storage_model_path="${storage_base_dir}/${model}" \
    --model_path="${model_base_dir}/${model}" \
    --output_dir="${output_base_dir}/${model}/qwen_sampling" \
    --data="${data_dir}" \
    --temperature=0.7 \
    --top_p=0.8 \
    --extra_body='{"chat_template_kwargs": {"enable_thinking": false}, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.05}'
done
