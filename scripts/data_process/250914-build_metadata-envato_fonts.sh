#!/bin/bash
set -e

slow_safe_pkill() {
    sleep 5
    local pattern="$1"
    local user="${2:-$USER}"

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

# [NOTE] change here for correct storage_cli path
# if storage_base_dir is not set, use default value
if [ -z "${storage_base_dir}" ]; then
    storage_base_dir=/home/vecglypher/mnt/
fi
tokenizer_path=workspace/hf_downloads/Qwen/Qwen3-4B


python -m src.svg_glyph_gen_v2_envato.extract_metadata \
    --file-metadata ../envato_fonts/metadata/fonts_file_level_metadata.csv \
    --zip-metadata ../envato_fonts/metadata/fonts_zip_level_metadata.csv \
    --output_dir data/processed_envato/metadata \
    --output-log-dir data/processed_envato/

python -m src.svg_glyph_gen_v2_envato.extract_zip_fonts \
    --zip-font-dir ../envato_fonts/zip_fonts \
    --metadata-dir data/processed_envato/metadata \
    --output-dir ../envato_fonts/fonts \
    --output-log-dir data/processed_envato/

python -m src.svg_glyph_gen_v2_envato.convert_to_gfont_format \
    -i data/processed_envato/metadata \
    -o data/processed_envato/metadata_in_gfont \
    -l data/processed_envato


python -m src.svg_glyph_gen_v2.filter_invalid_fonts \
    --input_metadata_jsonl data/processed_envato/metadata_in_gfont/ \
    --input_google_font_dir ../envato_fonts/fonts \
    --output_dir data/processed_envato/metadata_in_gfont-filter_invalid

python -m src.svg_glyph_gen_v2.render_pangram_for_fonts \
    --input_metadata_jsonl data/processed_envato/metadata_in_gfont-filter_invalid \
    --input_google_font_dir ../envato_fonts/fonts \
    --output_dir data/processed_envato/metadata_in_gfont-filter_invalid-pangram

python -m src.svg_glyph_gen_v2.render_pangram_for_fonts \
    --input_metadata_jsonl data/processed_envato/metadata_in_gfont-filter_invalid \
    --input_google_font_dir ../envato_fonts/fonts \
    --output_dir data/processed_envato/metadata_in_gfont-filter_invalid-pangram \
    --only_plot_hist &

python -m src.svg_glyph_gen_v2.filter_by_pangram_svg \
    --input_metadata_jsonl data/processed_envato/metadata_in_gfont-filter_invalid \
    --input_pangram_jsonl data/processed_envato/metadata_in_gfont-filter_invalid-pangram \
    --output_dir data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram

# Filter fonts by lmm ocr
output_base_dir=data/processed_envato/filter_fonts_by_lmm_ocr
mkdir -p ${output_base_dir}

content_file_dir=${output_base_dir}/content/
mkdir -p ${content_file_dir}
input_content_file=${content_file_dir}/content.txt

echo "GgAa" > ${input_content_file}

# optimize svg
input_metadata_jsonl=data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram
input_google_font_dir=../envato_fonts/fonts
output_dir=${output_base_dir}/normalized_svg/
python -m src.svg_glyph_gen_v2.batch_render_normalized_svg_v2 \
    --input_content_file $input_content_file \
    --input_metadata_jsonl $input_metadata_jsonl \
    --input_google_font_dir $input_google_font_dir \
    --num_workers 40 \
    --output_dir ${output_dir}


# build sft data
sft_base_dir=${output_base_dir}/sft/
input_svg_dir=${output_base_dir}/normalized_svg/
input_metadata_jsonl=data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram
python -m src.svg_glyph_gen_v2.build_sft_data_v2 \
    --input_svg_dir ${input_svg_dir} \
    --input_content_file ${input_content_file} \
    --input_metadata_jsonl ${input_metadata_jsonl} \
    --output_dir "${sft_base_dir}" \
    --output_log_dir "${sft_base_dir}" \
    --num_workers 40 \
    --chunk_size 500

python -m src.serve.decode_to_svg ${output_base_dir}/sft/ --num_workers 40 --batch_size 1000
python -m src.eval.svg2img_dir ${output_base_dir}/sft_decoded ${output_base_dir}/sft_decoded-img_base64-output --field output --width 192 --height 192
python -m src.eval.build_eval_data \
    --input_infer_jsonl_dir ${output_base_dir}/sft_decoded \
    --input_infer_img_base64_dir ${output_base_dir}/sft_decoded-img_base64-output \
    --field output

# Run lmm ocr eval
storage_weight_base_dir=workspace/hf_downloads/
local_weight_base_dir=saves/

function run_lmm_ocr_eval() {
    local model_url="$1"        # e.g. Qwen/Qwen2.5-VL-7B-Instruct
    local tag="$2"             # e.g. Qwen2.5-VL-7B-Instruct
    local dp="$3"              # data parallelism
    local tp="$4"              # tensor parallelism
    local dataset_dir="${output_base_dir}/ocr_eval_data"
    local eval_output_dir="${output_base_dir}/results_ocr_eval-${tag}"


    if [[ -f "${eval_output_dir}/DONE_INFER" ]]; then
        echo "skip lmm ocr: ${eval_output_dir}"
    else
        local storage_model_path="${storage_weight_base_dir}/${model_url}"

        local local_model_path="${local_weight_base_dir}/${model_url}"
        python scripts/tools/download_model_from_storage.py -i "${storage_model_path}" -o "${local_model_path}"

        export NCCL_DEBUG=WARN
        VLLM_HOST_IP=localhost VLLM_LOOPBACK_IP=localhost python src/serve/launch_server.py \
            "${local_model_path}" \
            --host "localhost" \
            --port 30000 \
            --data-parallel-address localhost \
            -dp ${dp} \
            -tp ${tp} \
            --max-model-len 12800 \
            --gpu-memory-utilization 0.8 \
            --limit-mm-per-prompt.video 0 \
            --limit-mm-per-prompt.image 1
            # --max_num_seqs 128 \

        mkdir -p ${eval_output_dir}
        curl http://localhost:30000/v1/models | tee ${eval_output_dir}/model_info.txt

        max_tokens=1024
        temperature=0.0
        python src/serve/api_infer.py \
            --data "${dataset_dir}" \
            --output_dir "${eval_output_dir}" \
            --model "${local_model_path}" \
            --base_url http://localhost:30000/v1 \
            --max_tokens "${max_tokens}" \
            --temperature "${temperature}" \
            # --top_p "${top_p}" \
            # --extra_body "${extra_body}"

        slow_safe_pkill "vllm"
        slow_safe_pkill "VLLM"
        slow_safe_pkill "multiprocessing.spawn"

        touch ${eval_output_dir}/DONE_INFER
    fi

    python -m src.eval.score_ocr_eval ${eval_output_dir} ${eval_output_dir}-acc-no_use_case --no_use_case
    python -m src.eval.score_ocr_eval ${eval_output_dir} ${eval_output_dir}-acc-use_case --use_case
    python -m src.eval.score_ocr_eval ${eval_output_dir} ${eval_output_dir}-acc-no_use_case-remove_whitespace --no_use_case --remove_whitespace
    python -m src.eval.score_ocr_eval ${eval_output_dir} ${eval_output_dir}-acc-use_case-remove_whitespace --use_case --remove_whitespace
}

# run_lmm_ocr_eval "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-7B-Instruct" 8 1
# run_lmm_ocr_eval "Qwen/Qwen2.5-VL-32B-Instruct" "Qwen2.5-VL-32B-Instruct" 4 2 # the performance is worst...
run_lmm_ocr_eval "Qwen/Qwen3-VL-30B-A3B-Instruct" "Qwen3-VL-30B-A3B-Instruct" 4 2
# run_lmm_ocr_eval "Qwen/Qwen3-VL-4B-Instruct" "Qwen3-VL-4B-Instruct" 8 1
# run_lmm_ocr_eval "Qwen/Qwen3-VL-8B-Instruct" "Qwen3-VL-8B-Instruct" 8 1


# gather results
find ${output_base_dir} -type f -name 'score_stats*.json' \
  -exec sh -c 'echo "=== {} ==="; grep accuracy "{}"; echo;' \; | tee ${output_base_dir}/score_stats.txt


# Use Qwen3-VL-30B-A3B-Instruct
python -m src.svg_glyph_gen_v2.filter_fonts_by_lmm_ocr \
    --input_gfont_metadata data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram \
    --input_lmm_ocr ${output_base_dir}/results_ocr_eval-Qwen3-VL-30B-A3B-Instruct-acc-use_case \
    --output_dir data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr


# NOTE: safe guard. We do not want to use it.
mv data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram data/processed_envato/.metadata_in_gfont-filter_invalid-filter_by_pangram

python -m src.svg_glyph_gen_v2.stat_field_values \
--input_metadata_jsonl data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr \
--output_dir data/processed_envato/metadata_in_gfont-stat_field_values

python -m src.svg_glyph_gen_v2.stat_font_vertical \
--input_gfont_metadata data/processed_envato/metadata_in_gfont-filter_invalid-filter_by_pangram-filtered_by_lmm_ocr \
--output_dir data/processed_envato/metadata_in_gfont-stat_font_vertical \
--input_google_font_dir ../envato_fonts/fonts

wait # there is a background process
echo -e "\033[32mFinish building metadata of envato fonts\033[0m"
