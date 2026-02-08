#!/bin/bash
set -e


for arg in "$@"; do
    case $arg in
        --models=*|models=*)
            IFS=',' read -r -a models <<< "${arg#*=}"
            ;;
        --eval_on_cluster_name=*|eval_on_cluster_name=*)
            eval_on_cluster_name="${arg#*=}"
            ;;
        --data_dir=*|data_dir=*)
            data_dir="${arg#*=}"
            ;;
        --tp=*|tp=*)
            tp="${arg#*=}"
            ;;
        --dp=*|dp=*)
            dp="${arg#*=}"
            ;;
        --output_base_dir=*|output_base_dir=*)
            output_base_dir="${arg#*=}"
            ;;
        --storage_mount_dir=*|storage_mount_dir=*)
            storage_mount_dir="${arg#*=}"
            ;;
        --storage_base_dir=*|storage_base_dir=*)
            storage_base_dir="${arg#*=}"
            ;;
    esac
done

if [[ -z $models ]]; then
    echo -e "\033[0;31mPlease provide a list of models to evaluate.\033[0m"
    exit 1
fi
if [[ -z $eval_on_cluster_name ]]; then
    echo -e "\033[0;31mPlease provide a name for the eval on CLUSTER.\033[0m"
    exit 1
fi


# script start
if [[ -z $output_base_dir ]]; then
    output_base_dir="/mnt/workspace/svg_glyph_llm/outputs/${eval_on_cluster_name}"
fi
if [[ -z $storage_base_dir ]]; then
    storage_base_dir=workspace/svg_glyph_llm/saves
fi
# CLUSTER does not allow writing to package dir: https://www.internal.example.com/tasks/?t=230671987
model_base_dir=/tmp/saves
if [[ -z $data_dir ]]; then
    data_dir=/mnt/workspace/svg_glyph_llm/data/processed/filtered_sft/250903-alphanumeric/ood_font_family_decon
fi
if [[ -z $storage_mount_dir ]]; then
    storage_mount_dir=/mnt/workspace/hf_downloads
fi
storage_ocr_model_dir=workspace/hf_downloads
ocr_model=Qwen/Qwen3-VL-30B-A3B-Instruct
if [[ -z $tp ]]; then
    tp=1
fi
if [[ -z $dp ]]; then
    dp=8
fi

# NOTE: model name, change to your own
# models=(
# 251015-ablate-lr_bs-num_epochs-gemma3_27b/bs_128-lr_1e-5
# 251015-ablate-lr_bs-num_epochs-gemma3_27b/bs_128-lr_2e-5
# 251015-ablate-lr_bs-num_epochs-gemma3_27b/bs_32-lr_1e-5
# 251015-ablate-lr_bs-num_epochs-gemma3_27b/bs_32-lr_1e-5-repeat_1
# )
# model path = storage_base_dir + model
echo -e "\033[0;32m===== Eval on CLUSTER CLI =======\033[0m"
for model in "${models[@]}"; do
    echo "model: $model"
done
echo "eval_on_cluster_name: $eval_on_cluster_name"
echo "data_dir: $data_dir"
echo "tp: $tp"
echo "dp: $dp"
echo "output_base_dir: $output_base_dir"
echo "storage_mount_dir: $storage_mount_dir"
echo -e "\033[0;32m===== Eval on CLUSTER CLI =======\033[0m"


for model in "${models[@]}"; do
bash scripts/eval_locally/eval_suite-template.sh \
    --storage_model_path="${storage_base_dir}/${model}" \
    --model_path="${model_base_dir}/${model}" \
    --output_dir="${output_base_dir}/${model}/t_0_0" \
    --data="${data_dir}" \
    --temperature=0.0 \
    --storage_mount_dir="${storage_mount_dir}" \
    --storage_ocr_model_path="${storage_ocr_model_dir}/${ocr_model}" \
    --ocr_model_path="${model_base_dir}/${ocr_model}" \
    --tp="${tp}" \
    --dp="${dp}"
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
    --ocr_model_path="${model_base_dir}/${ocr_model}" \
    --tp="${tp}" \
    --dp="${dp}"
done


exit

# example
eval_on_cluster_name=eval-251014-data-envato
models=\'251015-ablate-lr_bs-num_epochs-gemma3_27b/bs_128-lr_1e-5,251015-ablate-lr_bs-num_epochs-gemma3_27b/bs_128-lr_2e-5,251015-ablate-lr_bs-num_epochs-gemma3_27b/bs_32-lr_1e-5,251015-ablate-lr_bs-num_epochs-gemma3_27b/bs_32-lr_1e-5-repeat_1\'
tp=2
dp=4

python scripts/sft_on_cluster_submitter/submit.py \
-cp conf_eval \
base_args._cluster_param.config_file=scripts/eval_on_cluster/eval_on_cluster_cli.sh \
base_args._exp_name="${eval_on_cluster_name}" \
+base_args.models="${models}" \
+base_args.eval_on_cluster_name="${eval_on_cluster_name}" \
+base_args.dp="${dp}" \
+base_args.tp="${tp}"
# base_args._cluster_param.host=gpu_80g_pool # gpu_any_pool, gpu_pool_gt, gpu_80g_ib, gpu_80g_alt, gpu_a100_pool \
# base_args._cluster_param.scheduler_args=\'conda_pkg_id=REDACTED,clusterOncall=REDACTED,resourceAttribution=REDACTED,tags=REDACTED,modelTypeName=REDACTED\' \
# dry_run=true
