#!/bin/bash
# bash scripts/eval_locally/eval_suite-template.sh \
#     --storage_model_path=workspace/hf_downloads/Qwen/Qwen3 \
#     --model_path=saves/Qwen/Qwen3 \
#     --output_dir=outputs/Qwen/Qwen3 \
#     --data=/home/vecglypher/mnt/workspace/svg_glyph_llm/data/processed/filtered_sft/250903-alphanumeric/ood_font_family_decon \
#     --dry-run
set -e

# NOTE: Avoid NCCL INFO flood
export NCCL_DEBUG=WARN

# Check if the required number of arguments is passed
# We hardcode the command and request 8 GPUs per node.
# num_gpus="$(nvidia-smi -L | wc -l)"
# NOTE: no nvidia-smi on CLUSTER
num_gpus="$(gpustat --no-header  | wc -l)"
if [ $num_gpus -ne 8 ]; then
    echo "num_gpus: $num_gpus, expected 8, exiting..."
    exit 1
else
    echo "num_gpus: $num_gpus, check passed!"
fi

STORAGE_MODEL_PATH="workspace/hf_downloads/Qwen/Qwen3-4B-Instruct-2507"
MODEL_PATH="saves/Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_DIR="outputs/Qwen/Qwen3-4B-Instruct-2507"
DATA="/home/vecglypher/mnt/workspace/svg_glyph_llm/data/processed/filtered_sft/250903-alphanumeric/ood_font_family_decon"
DRY_RUN=false
MAX_TOKENS=3000
TEMPERATURE=0.0
TOP_P=1.0
EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0}'
DP=8
TP=1
STORAGE_MOUNT_DIR=/home/vecglypher/mnt/workspace/hf_downloads
STORAGE_OCR_MODEL_PATH="workspace/hf_downloads/Qwen/Qwen3-VL-30B-A3B-Instruct"
OCR_MODEL_PATH="saves/Qwen/Qwen3-VL-30B-A3B-Instruct"
# NOTE: problem of missing mm cache issue, turn it off.
# https://github.com/vllm-project/vllm/issues/26195?utm_source=chatgpt.com
MM_PROCESSOR_CACHE_GB=0
for arg in "$@"; do
    case $arg in
        --storage_model_path=*)
            STORAGE_MODEL_PATH="${arg#*=}"
            ;;
        --model_path=*)
            MODEL_PATH="${arg#*=}"
            ;;
        --output_dir=*)
            OUTPUT_DIR="${arg#*=}"
            ;;
        --data=*)
            DATA="${arg#*=}"
            ;;
        --dry_run)
            DRY_RUN=true
            ;;
        --max_tokens=*)
            MAX_TOKENS="${arg#*=}"
            ;;
        --temperature=*)
            TEMPERATURE="${arg#*=}"
            ;;
        --top_p=*)
            TOP_P="${arg#*=}"
            ;;
        --extra_body=*)
            EXTRA_BODY="${arg#*=}"
            ;;
        --dp=*)
            DP="${arg#*=}"
            ;;
        --tp=*)
            TP="${arg#*=}"
            ;;
        --storage_mount_dir=*)
            STORAGE_MOUNT_DIR="${arg#*=}"
            ;;
        --storage_ocr_model_path=*)
            STORAGE_OCR_MODEL_PATH="${arg#*=}"
            ;;
        --ocr_model_path=*)
            OCR_MODEL_PATH="${arg#*=}"
            ;;
        --mm_processor_cache_gb=*)
            MM_PROCESSOR_CACHE_GB="${arg#*=}"
            ;;
    esac
done

echo -e "\033[1;32m====== Start eval suite ======\033[0m"
echo -e "\033[1;36mSTORAGE_MODEL_PATH: ${STORAGE_MODEL_PATH}\033[0m"
echo -e "\033[1;35mMODEL_PATH: ${MODEL_PATH}\033[0m"
echo -e "\033[1;34mOUTPUT_DIR: ${OUTPUT_DIR}\033[0m"
echo -e "\033[1;36mDATA: ${DATA}\033[0m"
echo -e "\033[1;35mMAX_TOKENS: ${MAX_TOKENS}\033[0m"
echo -e "\033[1;34mTEMPERATURE: ${TEMPERATURE}\033[0m"
echo -e "\033[1;36mTOP_P: ${TOP_P}\033[0m"
echo -e "\033[1;35mEXTRA_BODY: ${EXTRA_BODY}\033[0m"
echo -e "\033[1;34mDP: ${DP}\033[0m"
echo -e "\033[1;36mTP: ${TP}\033[0m"
echo -e "\033[1;35mSTORAGE_MOUNT_DIR: ${STORAGE_MOUNT_DIR}\033[0m"
echo -e "\033[1;34mSTORAGE_OCR_MODEL_PATH: ${STORAGE_OCR_MODEL_PATH}\033[0m"
echo -e "\033[1;36mOCR_MODEL_PATH: ${OCR_MODEL_PATH}\033[0m"
echo -e "\033[1;35mMM_PROCESSOR_CACHE_GB: ${MM_PROCESSOR_CACHE_GB}\033[0m"
if [ "${DRY_RUN}" = true ]; then
    echo -e "\033[1;33mDRY_RUN: exiting\033[0m"
    exit
fi


slow_safe_pkill() {
    sleep 5
    local pattern="$1"
    local user="${2:-"$(whoami)"}"

    # 1. Try TERM
    pkill -u "$user" -f -TERM "$pattern" 2>/dev/null || true
    sleep 5

    # 2. If still alive, try INT (like Ctrl+C)
    if pgrep -u "$user" -f "$pattern" >/dev/null; then
        pkill -u "$user" -f -INT "$pattern" 2>/dev/null || true
        sleep 5
    fi

    # 3. If still alive, force KILL
    if pgrep -u "$user" -f "$pattern" >/dev/null; then
        pkill -u "$user" -f -9 "$pattern" 2>/dev/null || true
        sleep 5
    fi
}


# setup flag dir
output_dir="${OUTPUT_DIR}"
output_flag_dir="${output_dir}/flag"
mkdir -p "${output_flag_dir}"

done_infer_flag="${output_flag_dir}/done_infer"
done_point_cloud_eval_flag="${output_flag_dir}/done_point_cloud_eval"
done_ocr_eval_flag="${output_flag_dir}/done_ocr_eval"
done_img_eval_flag="${output_flag_dir}/done_img_eval"
done_eval_flag="${output_flag_dir}/done_eval"

if [ -f "${done_eval_flag}" ]; then
    echo -e "\033[1;32m${output_dir} already finished, skipping\033[0m"
    exit
fi

# start infer
if [ -f "${done_infer_flag}" ]; then
    echo -e "\033[1;32m${output_dir} infer already finished, skipping infer\033[0m"
else
    # download model
    storage_model_path="${STORAGE_MODEL_PATH}"
    model_path="${MODEL_PATH}"
    python scripts/tools/download_model_from_storage.py -i "${storage_model_path}" -o "${model_path}"

    data="${DATA}"
    dp="${DP}"
    tp="${TP}"
    VLLM_HOST_IP=localhost VLLM_LOOPBACK_IP=localhost python src/serve/launch_server.py \
        "${model_path}" \
        --host "localhost" \
        --port 30000 \
        --data-parallel-address localhost \
        --gpu-memory-utilization 0.8 \
        --mm-processor-cache-gb ${MM_PROCESSOR_CACHE_GB} \
        -dp ${dp} \
        -tp ${tp}

    curl http://localhost:30000/v1/models | tee "${output_dir}"/model_info.txt

    max_tokens="${MAX_TOKENS}"
    temperature="${TEMPERATURE}"
    top_p="${TOP_P}"
    extra_body="${EXTRA_BODY}"
    python src/serve/api_infer.py \
        --data "${data}" \
        --output_dir "${output_dir}"/infer \
        --model "${model_path}" \
        --base_url http://localhost:30000/v1 \
        --max_tokens "${max_tokens}" \
        --temperature "${temperature}" \
        --top_p "${top_p}" \
        --extra_body "${extra_body}"
    slow_safe_pkill "vllm"
    slow_safe_pkill "VLLM"
    slow_safe_pkill "multiprocessing.spawn"
    # nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv | column -t -s ",";
    gpustat

    touch "${done_infer_flag}"
fi


# generate images for ocr and img eval
if [[ -f "${done_ocr_eval_flag}" ]] && [[ -f "${done_img_eval_flag}" ]]; then
    echo -e "\033[1;32m${output_dir} ocr and img eval already finished, skipping generating images\033[0m"
else
    python -m src.serve.decode_to_svg "${output_dir}"/infer
    python -m src.eval.svg2img_dir "${output_dir}"/infer_decoded "${output_dir}"/infer_decoded-img_base64-predict --field predict --width 192 --height 192
    python -m src.eval.svg2img_dir "${output_dir}"/infer_decoded "${output_dir}"/infer_decoded-img_base64-label --field label --width 192 --height 192
    python -m src.eval.build_eval_data --input_infer_jsonl_dir "${output_dir}"/infer_decoded --input_infer_img_base64_dir "${output_dir}"/infer_decoded-img_base64-predict --field predict
fi

# start ocr eval
if [[ -f "${done_ocr_eval_flag}" ]]; then
    echo -e "\033[1;32m${output_dir} ocr eval already finished, skipping ocr eval\033[0m"
else
    storage_ocr_model_path="${STORAGE_OCR_MODEL_PATH}"
    ocr_model_path="${OCR_MODEL_PATH}"
    python scripts/tools/download_model_from_storage.py -i "${storage_ocr_model_path}" -o "${ocr_model_path}"

    VLLM_HOST_IP=localhost VLLM_LOOPBACK_IP=localhost python src/serve/launch_server.py \
        "${ocr_model_path}" \
        --host "localhost" \
        --port 30000 \
        --data-parallel-address localhost \
        -dp 4 \
        -tp 2 \
        --max-model-len 12800 \
        --gpu-memory-utilization 0.8 \
        --limit-mm-per-prompt.video 0 \
        --limit-mm-per-prompt.image 1 \
        --mm-encoder-tp-mode data
        # --max_num_seqs 128 \

    curl http://localhost:30000/v1/models | tee "${output_dir}"/eval_model_info.txt

    # qwen2.5-vl-7b-instruct dp=8 tp=1
    python src/serve/api_infer.py \
        --data "${output_dir}"/ocr_eval_data \
        --output_dir "${output_dir}"/ocr_infer \
        --model "${ocr_model_path}" \
        --base_url http://localhost:30000/v1 \
        --max_tokens 256 \
        --temperature 0.0
    slow_safe_pkill "vllm"
    slow_safe_pkill "VLLM"
    slow_safe_pkill "multiprocessing.spawn"
    # nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv | column -t -s ",";
    gpustat

    python -m src.eval.score_ocr_eval "${output_dir}"/ocr_infer "${output_dir}"/results_ocr_eval-no_use_case --no_use_case
    python -m src.eval.score_ocr_eval "${output_dir}"/ocr_infer "${output_dir}"/results_ocr_eval-use_case --use_case

    touch "${done_ocr_eval_flag}"
fi


# start img eval
if [[ -f "${done_img_eval_flag}" ]]; then
    echo -e "\033[1;32m${output_dir} img eval already finished, skipping img eval\033[0m"
else
    storage_vgg_model_path="${STORAGE_MOUNT_DIR}/eval_ckpts/vgg16-397923af.pth"
    vgg_model_path="$HOME/.cache/torch/hub/checkpoints/vgg16-397923af.pth"
    if [[ ! -f "${vgg_model_path}" ]]; then
        echo "${vgg_model_path} does not exist, downloading from ${storage_vgg_model_path}"
        mkdir -p "$(dirname "${vgg_model_path}")"
        storage_cli get "${storage_vgg_model_path}" "${vgg_model_path}" --threads 20
    else
        echo "${vgg_model_path} already exists, skipping download"
    fi

    HF_DINO_MODEL_PATH=${STORAGE_MOUNT_DIR}/facebook/dinov2-base \
    HF_CLIP_MODEL_PATH=${STORAGE_MOUNT_DIR}/openai/clip-vit-base-patch32 \
    TORCH_HUB_CKPT_DIR=${STORAGE_MOUNT_DIR}/eval_ckpts \
    OPENAI_CLIP_CACHE_DIR=${STORAGE_MOUNT_DIR}/eval_ckpts \
    CUDA_VISIBLE_DEVICES=0 \
    python -m src.eval.score_img_eval \
        "${output_dir}"/infer_decoded  \
        "${output_dir}"/infer_decoded-img_base64-predict \
        "${output_dir}"/infer_decoded-img_base64-label \
        "${output_dir}"/results_img_eval

    slow_safe_pkill "score_img_eval"
    # nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv | column -t -s ",";
    gpustat

    touch "${done_img_eval_flag}"
fi


# start point cloud eval
if [[ -f "${done_point_cloud_eval_flag}" ]]; then
    echo -e "\033[1;32m${output_dir} point cloud eval already finished, skipping point cloud\033[0m"
else
    # NOTE: May encounter error: A100 compiled pytorch3d, does not work on H100 due to the provided PTX was compiled with an unsupported toolchain.
    set +e

    python -m src.eval.score_point_cloud_eval \
        --input_svg_dir ${output_dir}/infer \
        --output_dir ${output_dir}/results_point_cloud_eval

    status=$?
    if [ $status -ne 0 ]; then
        echo "point cloud eval failed, retrying with --device cpu. No alignment and scale estimation"

        python -m src.eval.score_point_cloud_eval \
            --input_svg_dir ${output_dir}/infer \
            --output_dir ${output_dir}/results_point_cloud_eval \
            --device cpu
    else
        echo "point cloud eval succeeded, running with --align_pcd and --estimate_scale on cuda"
        # NOTE: turn off alignment, as the results are similar after alignment
        python -m src.eval.score_point_cloud_eval \
            --input_svg_dir ${output_dir}/infer \
            --output_dir ${output_dir}/results_point_cloud_eval-align_pcd \
            --align_pcd

        python -m src.eval.score_point_cloud_eval \
            --input_svg_dir ${output_dir}/infer \
            --output_dir ${output_dir}/results_point_cloud_eval-align_pcd-estimate_scale \
            --align_pcd \
            --estimate_scale
    fi
    set -e

    touch "${done_point_cloud_eval_flag}"
fi


touch "${done_eval_flag}"
echo -e "\033[1;32mDONE\033[0m"
