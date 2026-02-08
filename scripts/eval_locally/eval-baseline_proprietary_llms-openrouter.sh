#!/bin/bash
max_tokens=3000

# temperature=0.0
# top_p=1.0
# extra_body='{"chat_template_kwargs": {"enable_thinking": false}, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0}'

temperature=0.7
top_p=0.8
extra_body='{"chat_template_kwargs": {"enable_thinking": false}, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.05}'

data="/home/vecglypher/mnt/workspace/svg_glyph_llm/data/processed/filtered_sft/250903-alphanumeric/ood_font_family_decon"

base_url=https://openrouter.ai/api/v1

model_path_list=(
anthropic/claude-sonnet-4.5
openai/gpt-5-nano
openai/gpt-5-mini
google/gemini-2.5-flash
anthropic/claude-haiku-4.5
openai/gpt-5
google/gemini-2.5-pro
)
output_base_dir=outputs/baseline_proprietary_llms/

max_samples=5000

for model_path in "${model_path_list[@]}"; do
    output_dir=${output_base_dir}/${model_path}

    python src/serve/api_infer.py \
        --data "${data}" \
        --output_dir "${output_dir}"/infer \
        --model "${model_path}" \
        --base_url "${base_url}" \
        --max_tokens "${max_tokens}" \
        --temperature "${temperature}" \
        --top_p "${top_p}" \
        --extra_body "${extra_body}" \
        --max_samples "${max_samples}" \
        --reasoning_effort "minimal"
done

exit

# remove ``` and ```svg ```xml as `outputs/baseline_proprietary_llms-cleaned`

# NOTE: eval on cluster with interactive debugging
# run command to mount dir: source /packages/torchx_conda_mount/mount.sh
model_path_list=(
anthropic/claude-sonnet-4.5
openai/gpt-5-nano
openai/gpt-5-mini
google/gemini-2.5-flash
anthropic/claude-haiku-4.5
openai/gpt-5
google/gemini-2.5-pro
)
output_base_dir_cluster=/mnt/workspace/svg_glyph_llm/outputs/baseline_proprietary_llms-cleaned
for model_path in "${model_path_list[@]}"; do
    output_dir=${output_base_dir_cluster}/${model_path}
    bash scripts/eval_locally/eval_suite-template-specific_baselines.sh --output_dir="${output_dir}"
done
