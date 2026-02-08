#! /bin/bash
# [force_redownload=1] [skip_upload=1] [skip_download=1] bash scripts/tools/download_upload_weights.sh

mkdir -p data/hf_downloads

# qwen3 instruct (non-thinking): 4 / 30-a3
# qwen3 base: 4 / 30-a3
# qwen3-coder: 30-a3
# Llama 3.2 base: 3
# Llama 3.2 instruct: 3
# gemma 3 base: 4
# gemma 3 instruct: 4

model_urls=(
# qwen3
Qwen/Qwen3-4B-Instruct-2507
Qwen/Qwen3-30B-A3B-Instruct-2507
Qwen/Qwen3-Coder-30B-A3B-Instruct

Qwen/Qwen3-30B-A3B
Qwen/Qwen3-32B
Qwen/Qwen3-14B
Qwen/Qwen3-8B
Qwen/Qwen3-4B
Qwen/Qwen3-1.7B
Qwen/Qwen3-0.6B

Qwen/Qwen3-30B-A3B-Base
Qwen/Qwen3-14B-Base
Qwen/Qwen3-8B-Base
Qwen/Qwen3-4B-Base
Qwen/Qwen3-1.7B-Base
Qwen/Qwen3-0.6B-Base

# qwen2.5
Qwen/Qwen2.5-Coder-32B-Instruct
Qwen/Qwen2.5-Coder-32B

# gemma 3
google/gemma-3-27b-pt
google/gemma-3-27b-it
google/gemma-3-12b-pt
google/gemma-3-12b-it
google/gemma-3-4b-pt
google/gemma-3-4b-it
google/gemma-3-1b-pt
google/gemma-3-1b-it
google/gemma-3-270m-it
google/gemma-3-270m

# openai
openai/gpt-oss-20b

# llama
meta-llama/Llama-3.1-8B-Instruct
meta-llama/Llama-3.2-3B-Instruct
meta-llama/Llama-3.2-1B-Instruct
meta-llama/Llama-3.1-8B
meta-llama/Llama-3.2-3B
meta-llama/Llama-3.2-1B
meta-llama/Llama-3.3-70B-Instruct

# eval
# qwen 2.5 vl
Qwen/Qwen2.5-VL-7B-Instruct
openai/clip-vit-base-patch32
facebook/dinov2-base
Qwen/Qwen3-VL-30B-A3B-Instruct
Qwen/Qwen3-VL-8B-Instruct
Qwen/Qwen3-VL-4B-Instruct
)
storage_mnt_dir=/home/vecglypher/mnt
storage_base_dir=workspace/hf_downloads

if [[ -z $skip_download ]]; then
    for model_url in ${model_urls[@]}; do
        storage_dir=${storage_mnt_dir}/${storage_base_dir}/${model_url}
        local_dir=data/hf_downloads/${model_url}
        if [[ -n $force_redownload ]]; then
            echo -e "\033[31mForce redownload ${model_url}\033[0m"
        elif [[ -d ${storage_dir} ]]; then
            echo -e "\033[31mDownloaded ${model_url}\033[0m"

            continue
        else
            echo -e "\033[32mDownloading ${model_url}\033[0m"

        fi

        if HF_HUB_DISABLE_XET=1 with-proxy hf download \
        ${model_url} --repo-type model --local-dir ${local_dir} ; then
            echo -e "\033[32mSuccessfully downloaded ${local_dir}\033[0m"
        else
            echo -e "\033[31mFailed to download ${local_dir}\033[0m"
            rm -rf "${local_dir}"
            exit 1
        fi
    done
fi

if [[ -z $skip_upload ]]; then
    for model_url in ${model_urls[@]}; do
        storage_dir=${storage_mnt_dir}/${storage_base_dir}/${model_url}
        local_dir=data/hf_downloads/${model_url}
        if [[ -d ${storage_dir} ]]; then
            echo -e "\033[31mUploaded ${model_url}\033[0m"
            continue
        else
            echo -e "\033[32mUploading ${model_url}\033[0m"

        fi
        storage_cli  --prod-use-cython-client  mkdirs workspace/hf_downloads/${model_url}
        storage_cli  --prod-use-cython-client  putr ${local_dir} workspace/hf_downloads/${model_url} --threads 20 --jobs 10
        if [[ $? -ne 0 ]]; then
            echo -e "\033[31mFailed to upload ${storage_dir}\033[0m"
            rm -rf "${storage_dir}"
            echo -e "\033[31mRemoved ${storage_dir}\033[0m"
            exit 1
        fi
    done
fi
