#!/bin/bash
set -e

# ============

eval_on_cluster_name=eval-251117-ablate-lora-merged
models_array=(
251117-ablate-lora-merged/4b-rel_coord
251117-ablate-lora-merged/4b-abs_coord
251117-ablate-lora-merged/27b-rel_coord
251117-ablate-lora-merged/27b-abs_coord
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
