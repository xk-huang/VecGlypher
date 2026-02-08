#!/bin/bash
set -e

output_base_dir=outputs/250919-eval-gfont-baselines
storage_base_dir=workspace/hf_downloads
model_base_dir=saves
data_dir=/home/vecglypher/mnt/workspace/svg_glyph_llm/data/legacy/251009-processed/filtered_sft/250903-alphanumeric/ood_font_family

models=(
google/gemma-3-27b-it
google/gemma-3-12b-it
google/gemma-3-4b-it

meta-llama/Llama-3.1-8B-Instruct
meta-llama/Llama-3.2-3B-Instruct

Qwen/Qwen3-32B
Qwen/Qwen2.5-Coder-32B-Instruct
# 30B MoE models leads to OOM for tp=2, dp=4
# Qwen/Qwen3-30B-A3B-Instruct-2507
# Qwen/Qwen3-Coder-30B-A3B-Instruct
Qwen/Qwen3-8B
Qwen/Qwen3-4B
)
for model in "${models[@]}"; do
    echo "Eval ${model}..."
done

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
