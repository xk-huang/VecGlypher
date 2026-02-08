# Model

Download weights

```bash
models=(Qwen/Qwen2.5-VL-7B-Instruct
Qwen/Qwen2.5-VL-32B-Instruct
Qwen/Qwen3-4B
Qwen/Qwen3-VL-4B-Instruct
Qwen/Qwen3-VL-8B-Instruct
Qwen/Qwen3-VL-30B-A3B-Instruct
google/gemma-3-27b-it
google/gemma-3-4b-it
meta-llama/Llama-3.3-70B-Instruct
)
for model in ${models[@]}; do
    hf download "$model" --repo-type model --local-dir "saves/$model"
done
```