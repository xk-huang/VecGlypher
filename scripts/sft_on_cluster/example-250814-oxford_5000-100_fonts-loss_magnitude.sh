#!/usr/bin/sh

# --scheduler_args "conda_pkg_id=REDACTED"
# torchx run cluster_scripts/cluster.py:train \
#     --h tc_any \
#     --nnodes=1 --nproc_per_node=8 \
#     --max_retries 0 \
#     --script './cluster_scripts/setup_and_run_python.sh' \
#     -- \
#     src/sft/train.py \
#     sft/configs/train/a_z-300_fonts-no_style-qwen3_4b-full_sft.yaml \
#     output_dir=/mnt/workspace/llama-factory-saves/$(date '+%y%m%d')/a_z-300_fonts-no_style-qwen3_4b-full_sft-'${app_id}'

# interactive debugging:
# TORCHX_INTERACTIVE=1 bash cluster_scripts/launch_cluster_param_gpu.sh
# mint shell
# cd /packages/metaconda_demo && sh $RUN_INTERACTIVE
# mint exec 'cd /packages/metaconda_demo && sh $RUN_INTERACTIVE'
# mint exec 'export PATH=/packages/torchx_base_conda_env/bin:$PATH && cd /packages/metaconda_demo  && sh $RUN_INTERACTIVE'

# tc_any + --nproc_per_node=1 = tc_any with 1 GPU

set -e

# 8b
# char and word
nnodes=1
nproc_per_node=8
max_retries=5

config_file=src/sft/configs/train/qwen3_8b-full_sft.yaml

_exp_dir_name=250814-oxford_5000-100_fonts-loss_magnitude
_exp_name=char_and_word
output_dir=/mnt/workspace/svg_glyph_llm/saves/${_exp_dir_name}/${_exp_name}

dataset_dir=/mnt/workspace/svg_glyph_llm/data/processed/sft/250813-oxford_5000-100_fonts/
dataset=train-sample_100,train-alphanumeric
eval_dataset=ood_test-sample_30-contents_600,ood_test-alphanumeric

per_device_train_batch_size=2
gradient_accumulation_steps=2

learning_rate=5.0e-5
weight_decay=0.01
num_train_epochs=5.0

save_strategy=steps
save_steps=500
cutoff_len=6144

torchx run cluster_scripts/cluster.py:train \
    --h tc_any \
    --nnodes=${nnodes} --nproc_per_node=${nproc_per_node} \
    --max_retries ${max_retries} \
    --script './cluster_scripts/setup_and_run_python.sh' \
    -- \
    src/sft/train.py \
    $config_file \
    output_dir=${output_dir} \
    dataset_dir=${dataset_dir} \
    dataset=${dataset} \
    eval_dataset=${eval_dataset} \
    per_device_train_batch_size=${per_device_train_batch_size} \
    gradient_accumulation_steps=${gradient_accumulation_steps} \
    learning_rate=${learning_rate} \
    weight_decay=${weight_decay} \
    num_train_epochs=${num_train_epochs} \
    save_strategy=${save_strategy} \
    save_steps=${save_steps} \
    cutoff_len=${cutoff_len}


# char
nnodes=1
nproc_per_node=8
max_retries=5

config_file=src/sft/configs/train/qwen3_8b-full_sft.yaml

_exp_dir_name=250814-oxford_5000-100_fonts-loss_magnitude
_exp_name=char
output_dir=/mnt/workspace/svg_glyph_llm/saves/${_exp_dir_name}/${_exp_name}

dataset_dir=/mnt/workspace/svg_glyph_llm/data/processed/sft/250813-oxford_5000-100_fonts/
dataset=train-alphanumeric
eval_dataset=ood_test-alphanumeric

per_device_train_batch_size=2
gradient_accumulation_steps=2

learning_rate=5.0e-5
weight_decay=0.01
num_train_epochs=5.0

save_strategy=steps
save_steps=500
cutoff_len=6144

torchx run cluster_scripts/cluster.py:train \
    --h tc_any \
    --nnodes=${nnodes} --nproc_per_node=${nproc_per_node} \
    --max_retries ${max_retries} \
    --script './cluster_scripts/setup_and_run_python.sh' \
    -- \
    src/sft/train.py \
    $config_file \
    output_dir=${output_dir} \
    dataset_dir=${dataset_dir} \
    dataset=${dataset} \
    eval_dataset=${eval_dataset} \
    per_device_train_batch_size=${per_device_train_batch_size} \
    gradient_accumulation_steps=${gradient_accumulation_steps} \
    learning_rate=${learning_rate} \
    weight_decay=${weight_decay} \
    num_train_epochs=${num_train_epochs} \
    save_strategy=${save_strategy} \
    save_steps=${save_steps} \
    cutoff_len=${cutoff_len}

# word
nnodes=1
nproc_per_node=8
max_retries=5

config_file=src/sft/configs/train/qwen3_8b-full_sft.yaml

_exp_dir_name=250814-oxford_5000-100_fonts-loss_magnitude
_exp_name=word
output_dir=/mnt/workspace/svg_glyph_llm/saves/${_exp_dir_name}/${_exp_name}

dataset_dir=/mnt/workspace/svg_glyph_llm/data/processed/sft/250813-oxford_5000-100_fonts/
dataset=train-sample_100
eval_dataset=ood_test-sample_30-contents_600

per_device_train_batch_size=2
gradient_accumulation_steps=2

learning_rate=5.0e-5
weight_decay=0.01
num_train_epochs=5.0

save_strategy=steps
save_steps=500
cutoff_len=6144

torchx run cluster_scripts/cluster.py:train \
    --h tc_any \
    --nnodes=${nnodes} --nproc_per_node=${nproc_per_node} \
    --max_retries ${max_retries} \
    --script './cluster_scripts/setup_and_run_python.sh' \
    -- \
    src/sft/train.py \
    $config_file \
    output_dir=${output_dir} \
    dataset_dir=${dataset_dir} \
    dataset=${dataset} \
    eval_dataset=${eval_dataset} \
    per_device_train_batch_size=${per_device_train_batch_size} \
    gradient_accumulation_steps=${gradient_accumulation_steps} \
    learning_rate=${learning_rate} \
    weight_decay=${weight_decay} \
    num_train_epochs=${num_train_epochs} \
    save_strategy=${save_strategy} \
    save_steps=${save_steps} \
    cutoff_len=${cutoff_len}


exit

# local run test
learning_rate=5.0e-5
weight_decay=0.01
CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node 1 \
    --no-python \
    --standalone \
    -- \
    ./cluster_scripts/setup_and_run_python.sh \
    src/sft/train.py \
    src/sft/configs/train/qwen3_4b-full_sft.yaml \
    learning_rate=${learning_rate} \
    output_dir=saves/$(date '+%y%m%d')/qwen3_4b-full_sft-lr_${learning_rate}/ \
    dataset_dir=/mnt/workspace/svg_glyph_llm/data/processed/sft/250813-oxford_5000-100_fonts/  \
    dataset=train-sample_100,train-alphanumeric \
    eval_dataset=ood_test-sample_30-contents_600,ood_test-alphanumeric \
    per_device_train_batch_size=1 \
    weight_decay=${weight_decay} \
    model_name_or_path=/mnt/workspace/hf_downloads/Qwen/Qwen3-4B
