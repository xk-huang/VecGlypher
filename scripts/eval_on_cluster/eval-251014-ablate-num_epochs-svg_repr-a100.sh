#!/bin/bash
set -e

# models
# 251014-ablate-num_epochs-svg_repr/num_epochs_1-abs_coord
# 251014-ablate-num_epochs-svg_repr/num_epochs_3-abs_coord
# 251014-ablate-num_epochs-svg_repr/num_epochs_5-abs_coord
# 251014-ablate-num_epochs-svg_repr/num_epochs_1-rel_coord
# 251014-ablate-num_epochs-svg_repr/num_epochs_3-rel_coord
# 251014-ablate-num_epochs-svg_repr/num_epochs_5-rel_coord

eval_on_cluster_name=eval-251014-ablate-num_epochs-svg_repr-a100
models=\'251014-ablate-num_epochs-svg_repr/num_epochs_1-abs_coord,251014-ablate-num_epochs-svg_repr/num_epochs_3-abs_coord,251014-ablate-num_epochs-svg_repr/num_epochs_5-abs_coord,251014-ablate-num_epochs-svg_repr/num_epochs_1-rel_coord,251014-ablate-num_epochs-svg_repr/num_epochs_3-rel_coord,251014-ablate-num_epochs-svg_repr/num_epochs_5-rel_coord\'
tp=1
dp=8

python scripts/sft_on_cluster_submitter/submit.py \
-cp conf_eval \
base_args._cluster_param.config_file=scripts/eval_on_cluster/eval_on_cluster_cli.sh \
base_args._exp_name="${eval_on_cluster_name}" \
+base_args.models="${models}" \
+base_args.eval_on_cluster_name="${eval_on_cluster_name}" \
+base_args.dp="${dp}" \
+base_args.tp="${tp}" \
base_args._cluster_param.host=gpu_a100_pool
# base_args._cluster_param.scheduler_args=\'conda_pkg_id=REDACTED,clusterOncall=REDACTED,resourceAttribution=REDACTED,tags=REDACTED,modelTypeName=REDACTED\' \
# dry_run=true
