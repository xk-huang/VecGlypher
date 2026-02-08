#!/bin/bash
set -e

# models
# 251015-ablate-lr_bs/bs_128-lr_1e-5
# 251015-ablate-lr_bs/bs_128-lr_2e-5
# 251015-ablate-lr_bs/bs_32-lr_1e-5

eval_on_cluster_name=eval-251015-ablate-lr_bs
models=\'251015-ablate-lr_bs/bs_128-lr_1e-5,251015-ablate-lr_bs/bs_128-lr_2e-5,251015-ablate-lr_bs/bs_32-lr_1e-5\'
tp=1
dp=8

python scripts/sft_on_cluster_submitter/submit.py \
-cp conf_eval \
base_args._cluster_param.config_file=scripts/eval_on_cluster/eval_on_cluster_cli.sh \
base_args._exp_name="${eval_on_cluster_name}" \
+base_args.models="${models}" \
+base_args.eval_on_cluster_name="${eval_on_cluster_name}" \
+base_args.dp="${dp}" \
+base_args.tp="${tp}"
# base_args._cluster_param.host=gpu_80g_pool # gpu_any_pool, gpu_pool_gt, gpu_80g_ib, gpu_80g_alt, gpu_a100_pool
# base_args._cluster_param.scheduler_args=\'conda_pkg_id=REDACTED,clusterOncall=REDACTED,resourceAttribution=REDACTED,tags=REDACTED,modelTypeName=REDACTED\' \
# dry_run=true
