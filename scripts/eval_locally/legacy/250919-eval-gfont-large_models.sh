#!/bin/bash
set -e

output_base_dir=outputs/250919-eval-gfont-large_models
storage_base_dir=workspace/svg_glyph_llm/saves/legacy

model_base_dir=saves
data_dir=/home/vecglypher/mnt/workspace/svg_glyph_llm/data/legacy/251009-processed/filtered_sft/250903-alphanumeric/ood_font_family

models=(
# gfont-ft
250910-google_font-ablate_scale/Qwen3-14B-rel_coord
250910-google_font-ablate_scale/Qwen3-32B-rel_coord
250910-google_font-ablate_scale/Qwen3-32B-rel_coord-fix_bs
# gfont-ft
250917-envato_pt_google_ft/gemma-3-27b-it-wo_envato_pt-gfont_ft_5_epoch
250917-envato_pt_google_ft/gemma-3-27b-it-wo_envato_pt-gfont_ft_1_epoch
# envato-pt
250915-envato-upper_bound/Qwen3-32B
250915-envato-upper_bound/Qwen2_5-Coder-32B-Instruct
250915-envato-upper_bound/gemma-3-27b-it
# 250915-envato-upper_bound/Llama-3_3-70B-Instruct
# envato-pt gfont-ft
250917-envato_pt_google_ft/gemma-3-27b-it-envato_pt_1_epoch-gfont_ft_1_epoch
250917-envato_pt_google_ft/gemma-3-27b-it-envato_pt_1_epoch-gfont_ft_5_epoch
)

for model in "${models[@]}"; do
bash scripts/eval_locally/eval_suite-template.sh \
    --storage_model_path="${storage_base_dir}/${model}" \
    --model_path="${model_base_dir}/${model}" \
    --output_dir="${output_base_dir}/${model}/t_0_0" \
    --data="${data_dir}" \
    --temperature=0.0 \
    --tp=2 \
    --dp=4
done

for model in "${models[@]}"; do
bash scripts/eval_locally/eval_suite-template.sh \
    --storage_model_path="${storage_base_dir}/${model}" \
    --model_path="${model_base_dir}/${model}" \
    --output_dir="${output_base_dir}/${model}/t_0_7" \
    --data="${data_dir}" \
    --temperature=0.7 \
    --tp=2 \
    --dp=4
done

for model in "${models[@]}"; do
bash scripts/eval_locally/eval_suite-template.sh \
    --storage_model_path="${storage_base_dir}/${model}" \
    --model_path="${model_base_dir}/${model}" \
    --output_dir="${output_base_dir}/${model}/qwen_sampling" \
    --data="${data_dir}" \
    --temperature=0.7 \
    --top_p=0.8 \
    --extra_body='{"chat_template_kwargs": {"enable_thinking": false}, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.05}' \
    --tp=2 \
    --dp=4
done
