#!/bin/bash
set -e

# NOTE: save dir = output_base_dir + model
output_base_dir=/mnt/workspace/svg_glyph_llm/outputs/debug-eval_on_cluster-250919-eval-gfont-small_models
storage_base_dir=workspace/svg_glyph_llm/save/legacy
s
# CLUSTER cluster does not allow writing to package dir: https://www.internal.example.com/tasks/?t=230671987
model_base_dir=/tmp/saves
data_dir=/mnt/workspace/svg_glyph_llm/data/legacy/251009-processed/filtered_sft/250903-alphanumeric/ood_font_family
storage_mount_dir=/mnt
storage_ocr_model_dir=workspace/hf_downloads
ocr_model=Qwen/Qwen3-VL-30B-A3B-Instruct

# NOTE: model name, change to your own
models=(
#
250910-google_font-ablate_svg_repr/Qwen3-1_7B-rel_coord
)

for model in "${models[@]}"; do
bash scripts/eval_locally/eval_suite-template.sh \
    --storage_model_path="${storage_base_dir}/${model}" \
    --model_path="${model_base_dir}/${model}" \
    --output_dir="${output_base_dir}/${model}/t_0_0" \
    --data="${data_dir}" \
    --temperature=0.0 \
    --storage_mount_dir="${storage_mount_dir}" \
    --storage_ocr_model_path="${storage_ocr_model_dir}/${ocr_model}" \
    --ocr_model_path="${model_base_dir}/${ocr_model}"
done

for model in "${models[@]}"; do
bash scripts/eval_locally/eval_suite-template.sh \
    --storage_model_path="${storage_base_dir}/${model}" \
    --model_path="${model_base_dir}/${model}" \
    --output_dir="${output_base_dir}/${model}/t_0_7" \
    --data="${data_dir}" \
    --temperature=0.7 \
    --storage_mount_dir="${storage_mount_dir}" \
    --storage_ocr_model_path="${storage_ocr_model_dir}/${ocr_model}" \
    --ocr_model_path="${model_base_dir}/${ocr_model}"
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
    --storage_mount_dir="${storage_mount_dir}" \
    --storage_ocr_model_path="${storage_ocr_model_dir}/${ocr_model}" \
    --ocr_model_path="${model_base_dir}/${ocr_model}"
done


exit

eval_on_cluster_name=debug-eval_on_cluster-250919-eval-gfont-small_models

python scripts/sft_on_cluster_submitter/submit.py \
-cp conf_eval \
base_args._exp_name="${eval_on_cluster_name}" \
base_args._cluster_param.config_file=scripts/eval_on_cluster/"${eval_on_cluster_name}".sh
# base_args._cluster_param.host=gpu_80g_pool # gpu_any_pool, gpu_pool_gt, gpu_80g_ib, gpu_80g_alt, gpu_a100_pool
# dry_run=true
