#!/bin/bash
set -e

# models
# 251020-data-envato-then-google/gemma-3-27b-it-abs_coord-envato_fonts-num_epochs_1
# 251020-data-envato-then-google/gemma-3-27b-it-abs_coord-envato_fonts-num_epochs_3
# 251020-data-envato-then-google/gemma-3-27b-it-abs_coord-envato_fonts-num_epochs_5
# 251020-data-envato-then-google/gemma-3-27b-it-envato_fonts-num_epochs_1
# 251020-data-envato-then-google/gemma-3-27b-it-envato_fonts-num_epochs_3
# 251020-data-envato-then-google/gemma-3-27b-it-envato_fonts-num_epochs_5


# ============
eval_on_cluster_name=eval-251020-data-envato-then-google
models=251020-data-envato-then-google/gemma-3-27b-it-abs_coord-envato_fonts-num_epochs_1
tp=2
dp=4

python scripts/sft_on_cluster_submitter/submit.py \
-cp conf_eval \
base_args._cluster_param.config_file=scripts/eval_on_cluster/eval_on_cluster_cli.sh \
base_args._exp_name="${eval_on_cluster_name}" \
+base_args.models="${models}" \
+base_args.eval_on_cluster_name="${eval_on_cluster_name}" \
+base_args.dp="${dp}" \
+base_args.tp="${tp}" \
base_args._cluster_param.scheduler_args=\'conda_pkg_id=REDACTED,clusterOncall=REDACTED,resourceAttribution=REDACTED,tags=REDACTED,modelTypeName=REDACTED\'
# base_args._cluster_param.host=gpu_80g_pool # gpu_any_pool, gpu_pool_gt, gpu_80g_ib, gpu_80g_alt, gpu_a100_pool
# dry_run=true


eval_on_cluster_name=eval-251020-data-envato-then-google
models=251020-data-envato-then-google/gemma-3-27b-it-abs_coord-envato_fonts-num_epochs_3
tp=2
dp=4

python scripts/sft_on_cluster_submitter/submit.py \
-cp conf_eval \
base_args._cluster_param.config_file=scripts/eval_on_cluster/eval_on_cluster_cli.sh \
base_args._exp_name="${eval_on_cluster_name}" \
+base_args.models="${models}" \
+base_args.eval_on_cluster_name="${eval_on_cluster_name}" \
+base_args.dp="${dp}" \
+base_args.tp="${tp}" \
base_args._cluster_param.scheduler_args=\'conda_pkg_id=REDACTED,clusterOncall=REDACTED,resourceAttribution=REDACTED,tags=REDACTED,modelTypeName=REDACTED\'
# base_args._cluster_param.host=gpu_80g_pool # gpu_any_pool, gpu_pool_gt, gpu_80g_ib, gpu_80g_alt, gpu_a100_pool
# dry_run=true


eval_on_cluster_name=eval-251020-data-envato-then-google
models=251020-data-envato-then-google/gemma-3-27b-it-abs_coord-envato_fonts-num_epochs_5
tp=2
dp=4

python scripts/sft_on_cluster_submitter/submit.py \
-cp conf_eval \
base_args._cluster_param.config_file=scripts/eval_on_cluster/eval_on_cluster_cli.sh \
base_args._exp_name="${eval_on_cluster_name}" \
+base_args.models="${models}" \
+base_args.eval_on_cluster_name="${eval_on_cluster_name}" \
+base_args.dp="${dp}" \
+base_args.tp="${tp}" \
base_args._cluster_param.scheduler_args=\'conda_pkg_id=REDACTED,clusterOncall=REDACTED,resourceAttribution=REDACTED,tags=REDACTED,modelTypeName=REDACTED\'
# base_args._cluster_param.host=gpu_80g_pool # gpu_any_pool, gpu_pool_gt, gpu_80g_ib, gpu_80g_alt, gpu_a100_pool
# dry_run=true


# ============
# 251020-data-envato-then-google/gemma-3-27b-it-envato_fonts-num_epochs_1
# 251020-data-envato-then-google/gemma-3-27b-it-envato_fonts-num_epochs_3
# 251020-data-envato-then-google/gemma-3-27b-it-envato_fonts-num_epochs_5
eval_on_cluster_name=eval-251020-data-envato-then-google
models=251020-data-envato-then-google/gemma-3-27b-it-envato_fonts-num_epochs_1
tp=2
dp=4

python scripts/sft_on_cluster_submitter/submit.py \
-cp conf_eval \
base_args._cluster_param.config_file=scripts/eval_on_cluster/eval_on_cluster_cli.sh \
base_args._exp_name="${eval_on_cluster_name}" \
+base_args.models="${models}" \
+base_args.eval_on_cluster_name="${eval_on_cluster_name}" \
+base_args.dp="${dp}" \
+base_args.tp="${tp}" \
base_args._cluster_param.scheduler_args=\'conda_pkg_id=REDACTED,clusterOncall=REDACTED,resourceAttribution=REDACTED,tags=REDACTED,modelTypeName=REDACTED\'
# base_args._cluster_param.host=gpu_80g_pool # gpu_any_pool, gpu_pool_gt, gpu_80g_ib, gpu_80g_alt, gpu_a100_pool
# dry_run=true


eval_on_cluster_name=eval-251020-data-envato-then-google
models=251020-data-envato-then-google/gemma-3-27b-it-envato_fonts-num_epochs_3
tp=2
dp=4

python scripts/sft_on_cluster_submitter/submit.py \
-cp conf_eval \
base_args._cluster_param.config_file=scripts/eval_on_cluster/eval_on_cluster_cli.sh \
base_args._exp_name="${eval_on_cluster_name}" \
+base_args.models="${models}" \
+base_args.eval_on_cluster_name="${eval_on_cluster_name}" \
+base_args.dp="${dp}" \
+base_args.tp="${tp}" \
base_args._cluster_param.scheduler_args=\'conda_pkg_id=REDACTED,clusterOncall=REDACTED,resourceAttribution=REDACTED,tags=REDACTED,modelTypeName=REDACTED\'
# base_args._cluster_param.host=gpu_80g_pool # gpu_any_pool, gpu_pool_gt, gpu_80g_ib, gpu_80g_alt, gpu_a100_pool
# dry_run=true


eval_on_cluster_name=eval-251020-data-envato-then-google
models=251020-data-envato-then-google/gemma-3-27b-it-envato_fonts-num_epochs_5
tp=2
dp=4

python scripts/sft_on_cluster_submitter/submit.py \
-cp conf_eval \
base_args._cluster_param.config_file=scripts/eval_on_cluster/eval_on_cluster_cli.sh \
base_args._exp_name="${eval_on_cluster_name}" \
+base_args.models="${models}" \
+base_args.eval_on_cluster_name="${eval_on_cluster_name}" \
+base_args.dp="${dp}" \
+base_args.tp="${tp}" \
base_args._cluster_param.scheduler_args=\'conda_pkg_id=REDACTED,clusterOncall=REDACTED,resourceAttribution=REDACTED,tags=REDACTED,modelTypeName=REDACTED\'
# base_args._cluster_param.host=gpu_80g_pool # gpu_any_pool, gpu_pool_gt, gpu_80g_ib, gpu_80g_alt, gpu_a100_pool
# dry_run=true
