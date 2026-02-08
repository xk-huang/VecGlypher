#!/bin/bash
set -e

# models
# workspace/svg_glyph_llm/saves/251014-data-envato/Llama-3_3-70B-Instruct
# workspace/svg_glyph_llm/saves/251014-data-envato/Llama-3_3-70B-Instruct-abs_coord
# workspace/svg_glyph_llm/saves/251014-data-envato/gemma-3-27b-it
# workspace/svg_glyph_llm/saves/251014-data-envato/gemma-3-27b-it-abs_coord

eval_on_cluster_name=eval-251014-data-envato
models=\'251014-data-envato/Llama-3_3-70B-Instruct,251014-data-envato/Llama-3_3-70B-Instruct-abs_coord\'
tp=4
dp=2

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



eval_on_cluster_name=eval-251014-data-envato
models=\'251014-data-envato/gemma-3-27b-it,251014-data-envato/gemma-3-27b-it-abs_coord\'
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



# ============
# single model job

eval_on_cluster_name=eval-251014-data-envato
models=251014-data-envato/Llama-3_3-70B-Instruct
tp=4
dp=2

python scripts/sft_on_cluster_submitter/submit.py \
-cp conf_eval \
base_args._cluster_param.config_file=scripts/eval_on_cluster/eval_on_cluster_cli.sh \
base_args._exp_name="${eval_on_cluster_name}" \
+base_args.models="${models}" \
+base_args.eval_on_cluster_name="${eval_on_cluster_name}" \
+base_args.dp="${dp}" \
+base_args.tp="${tp}" \
base_args._cluster_param.scheduler_args=\'conda_pkg_id=REDACTED,clusterOncall=REDACTED,resourceAttribution=REDACTED,tags=REDACTED,modelTypeName=REDACTED\'


eval_on_cluster_name=eval-251014-data-envato
models=251014-data-envato/Llama-3_3-70B-Instruct-abs_coord
tp=4
dp=2

python scripts/sft_on_cluster_submitter/submit.py \
-cp conf_eval \
base_args._cluster_param.config_file=scripts/eval_on_cluster/eval_on_cluster_cli.sh \
base_args._exp_name="${eval_on_cluster_name}" \
+base_args.models="${models}" \
+base_args.eval_on_cluster_name="${eval_on_cluster_name}" \
+base_args.dp="${dp}" \
+base_args.tp="${tp}" \
base_args._cluster_param.scheduler_args=\'conda_pkg_id=REDACTED,clusterOncall=REDACTED,resourceAttribution=REDACTED,tags=REDACTED,modelTypeName=REDACTED\'


eval_on_cluster_name=eval-251014-data-envato
models=251014-data-envato/gemma-3-27b-it
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


eval_on_cluster_name=eval-251014-data-envato
models=251014-data-envato/gemma-3-27b-it-abs_coord
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
