#!/bin/bash
set -e

# models
# workspace/svg_glyph_llm/saves/251020-data-envato-then-google-llama_70b-lr_5e_6
# 251020-data-envato-then-google-llama_70b-lr_5e_6/Llama-3_3-70B-Instruct-abs_coord-envato_fonts-num_epochs_1
# 251020-data-envato-then-google-llama_70b-lr_5e_6/Llama-3_3-70B-Instruct-abs_coord-envato_fonts-num_epochs_3
# 251020-data-envato-then-google-llama_70b-lr_5e_6/Llama-3_3-70B-Instruct-envato_fonts-num_epochs_1
# 251020-data-envato-then-google-llama_70b-lr_5e_6/Llama-3_3-70B-Instruct-envato_fonts-num_epochs_3

# ============
eval_on_cluster_name=eval-251020-data-envato-then-google-llama_70b-lr_5e_6
models_array=(
251020-data-envato-then-google-llama_70b-lr_5e_6/Llama-3_3-70B-Instruct-abs_coord-envato_fonts-num_epochs_1
251020-data-envato-then-google-llama_70b-lr_5e_6/Llama-3_3-70B-Instruct-abs_coord-envato_fonts-num_epochs_3
251020-data-envato-then-google-llama_70b-lr_5e_6/Llama-3_3-70B-Instruct-envato_fonts-num_epochs_1
251020-data-envato-then-google-llama_70b-lr_5e_6/Llama-3_3-70B-Instruct-envato_fonts-num_epochs_3
)
tp=4
dp=2

for models in "${models_array[@]}"; do
    echo "models: ${models}"
    python scripts/sft_on_cluster_submitter/submit.py \
    -cp conf_eval \
    base_args._cluster_param.config_file=scripts/eval_on_cluster/eval_on_cluster_cli.sh \
    base_args._exp_name="${eval_on_cluster_name}" \
    +base_args.models="${models}" \
    +base_args.eval_on_cluster_name="${eval_on_cluster_name}" \
    +base_args.dp="${dp}" \
    +base_args.tp="${tp}" \
    base_args._cluster_param.host=gpu_a100_pool \
    base_args._cluster_param.scheduler_args=\'conda_pkg_id=REDACTED,clusterOncall=REDACTED,resourceAttribution=REDACTED,tags=REDACTED,modelTypeName=REDACTED\'
    # base_args._cluster_param.host=gpu_80g_pool # gpu_any_pool, gpu_pool_gt, gpu_80g_ib, gpu_80g_alt, gpu_a100_pool
    # dry_run=true
done
