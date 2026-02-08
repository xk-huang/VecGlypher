# Inference

For example of inference and evaluation, please refer to the [evaluation doc](evaluation.md).

## Interactive chat

```bash
llamafactory-cli chat src/sft/configs/inference/a-z-300_fonts-no_style-qwen3_4b-full_sft.yaml
```

## Batch inference with vLLM

Start a local server:

```bash
model_path=<model_name_or_path>
vllm serve ${model_path} --host 0.0.0.0 --port 30000 -tp 1 -dp 1 --enable-log-requests

curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: dummy-key" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "messages": [
      {"role": "user", "content": "Write a haiku about GPUs"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }' \
  | jq
```

Run batch inference against the OpenAI-compatible endpoint:

```bash
python src/serve/api_infer.py \
  --data <jsonl_or_dir> \
  --output_dir outputs/infer \
  --model <model_name_or_path> \
  --base_url http://localhost:30000/v1
```

## Prompt Example

System
```
You are a specialized vector glyph designer creating SVG path elements.

CRITICAL REQUIREMENTS:
- Each glyph must be a complete, self-contained element, in reading order of the given text.
- Terminate each element with a newline character
- Output ONLY valid SVG elements
```

Instruction:
```
Font design requirements: wordspace quality, humanist sans-serif, 600 weight, calm, sans-serif, competent, business, stiff, normal style
Text content: X
```

Example

```bash
model_path=saves/251026-data-envato-then-google-text_img_merged/gemma-3-27b-it-abs_coord-envato_fonts-num_epochs_3
vllm serve ${model_path} --host 0.0.0.0 --port 30000 -tp 1 -dp 1 --enable-log-requests

curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: dummy-key" \
  -d '{
    "model": "saves/251026-data-envato-then-google-text_img_merged/gemma-3-27b-it-abs_coord-envato_fonts-num_epochs_3",
    "messages": [
      {
        "role": "system",
        "content": "You are a specialized vector glyph designer creating SVG path elements.\n\nCRITICAL REQUIREMENTS:\n- Each glyph must be a complete, self-contained element, in reading order of the given text.\n- Terminate each element with a newline character\n- Output ONLY valid SVG elements"
      },
      {
        "role": "user",
        "content": "Font design requirements: wordspace quality, humanist sans-serif, 600 weight, calm, sans-serif, competent, business, stiff, normal style\nText content: a"
      }
    ],

    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.05,

    "chat_template_kwargs": {
      "enable_thinking": false
    },

    "max_tokens": 1024
  }' \
  | jq
```

## Visualization

```bash
streamlit run src/tools/sft_data_visualizer.py --server.port 8443
```
