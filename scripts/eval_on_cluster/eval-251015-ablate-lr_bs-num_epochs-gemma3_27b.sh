#!/bin/bash
set -e


eval_on_cluster_name=eval-251015-ablate-lr_bs-num_epochs-gemma3_27b
models_array=(
251015-ablate-lr_bs-num_epochs-gemma3_27b/bs_128-lr_1e-5
251015-ablate-lr_bs-num_epochs-gemma3_27b/bs_128-lr_2e-5
251015-ablate-lr_bs-num_epochs-gemma3_27b/bs_32-lr_1e-5
251015-ablate-lr_bs-num_epochs-gemma3_27b/bs_32-lr_1e-5-repeat_1
)
tp=2
dp=4

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


eval_on_cluster_name=eval-251015-ablate-lr_bs-num_epochs-gemma3_27b
models_array=(
251015-ablate-lr_bs-num_epochs-gemma3_27b/bs_32-lr_1e-5-num_epochs_1
251015-ablate-lr_bs-num_epochs-gemma3_27b/bs_32-lr_1e-5-num_epochs_3
)
tp=2
dp=4

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
