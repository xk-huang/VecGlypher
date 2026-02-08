# Training (SFT)

Edit a training config such as `src/sft/configs/train/qwen3_0_6b-full_sft.yaml`
and update:

- `model_name_or_path` (base model path or HF ID)
- `dataset_dir` (must contain `dataset_info.json`)
- `output_dir`

Run training:

```bash
llamafactory-cli train src/sft/configs/train/qwen3_0_6b-full_sft.yaml
```

## Example

```bash
python scripts/sft_on_cluster_submitter/submit.py \
    -cn 250910-google_font-ablate_svg_repr-rel_coord \
    local_run=true \
    dry_run=true \
    [jobs={ALL,Qwen3-4B-rel_coord}]

# Check misc/submitter_artifacts for launch scripts
# Change model_name_or_path, dataset_dir, output_dir
```
