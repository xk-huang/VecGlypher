#!/bin/bash
set -e

# ============

eval_on_cluster_name=eval-251030-data-google-text_img_merged-ref_img
models_array=(
251030-data-google-text_img_merged/gemma-3-27b-it-abs_coord-envato_fonts-num_epochs_3
251030-data-google-text_img_merged/gemma-3-27b-it-envato_fonts-num_epochs_3
251030-data-google-text_img_merged/gemma-3-4b-it-abs_coord-envato_fonts-num_epochs_3
251030-data-google-text_img_merged/gemma-3-4b-it-envato_fonts-num_epochs_3
)
tp=2
dp=4
# change to ref image dataset
data_dir=/mnt/workspace/svg_glyph_llm/data/250903-alphanumeric-ref_img-b64_pil/ood_font_family_decon

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
    +base_args.data_dir="${data_dir}" \
    base_args._cluster_param.host=gpu_a100_pool \
    base_args._cluster_param.scheduler_args=\'conda_pkg_id=REDACTED,clusterOncall=REDACTED,resourceAttribution=REDACTED,tags=REDACTED,modelTypeName=REDACTED\'
    # base_args._cluster_param.host=gpu_80g_pool # gpu_any_pool, gpu_pool_gt, gpu_80g_ib, gpu_80g_alt, gpu_a100_pool
    # dry_run=true
done
