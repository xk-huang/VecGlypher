# Evaluation

Decode SVG outputs, render to images, and compute metrics:

```bash
python -m src.serve.decode_to_svg outputs/infer
python -m src.eval.svg2img_dir outputs/infer_decoded outputs/infer_decoded-img_base64-predict --field predict --width 192 --height 192
python -m src.eval.svg2img_dir outputs/infer_decoded outputs/infer_decoded-img_base64-label --field label --width 192 --height 192
python -m src.eval.score_img_eval outputs/infer_decoded outputs/infer_decoded-img_base64-predict outputs/infer_decoded-img_base64-label outputs/results_img_eval
```

## Prequisites

```bash
mkdir -p data/hf_downloads/eval_ckpts

wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt -O data/hf_downloads/eval_ckpts/ViT-B-32.pt
wget https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth -O data/hf_downloads/eval_ckpts/pt_inception-2015-12-05-6726825d.pth
wget https://download.pytorch.org/models/vgg16-397923af.pth -O data/hf_downloads/eval_ckpts/vgg16-397923af.pth
rsync -avP data/hf_downloads/eval_ckpts/vgg16-397923af.pth $HOME/.cache/torch/hub/checkpoints/vgg16-397923af.pth

# HF_HUB_DISABLE_XET=1
hf download openai/clip-vit-base-patch32 --repo-type model --local-dir data/hf_downloads/openai/clip-vit-base-patch32
hf download facebook/dinov2-base --repo-type model --local-dir data/hf_downloads/facebook/dinov2-base
```

## Example

Refer to `scripts/eval_on_cluster/eval_on_cluster_cli.sh` for detailed commands.

```bash
storage_model_path=_
model_base_dir=saves/
model=debug/250910-google_font-ablate_svg_repr-rel_coord/Qwen3-4B-rel_coord
output_base_dir=outputs/
data=data/processed/filtered_sft/250903-alphanumeric/ood_font_family_decon
storage_mount_dir=data/hf_downloads
storage_ocr_model_path=_
ocr_model=Qwen/Qwen3-VL-30B-A3B-Instruct
tp=1
dp=8

# Greedy decoding
bash scripts/eval_locally/eval_suite-template.sh \
    --storage_model_path="${storage_model_path}" \
    --model_path="${model_base_dir}/${model}" \
    --output_dir="${output_base_dir}/${model}/t_0_0" \
    --data="${data}" \
    --temperature=0.0 \
    --storage_mount_dir="${storage_mount_dir}" \
    --storage_ocr_model_path="${storage_ocr_model_path}" \
    --ocr_model_path="${model_base_dir}/${ocr_model}" \
    --tp="${tp}" \
    --dp="${dp}"

# Sampling decoding with qwen sampling settings
bash scripts/eval_locally/eval_suite-template.sh \
    --storage_model_path="${storage_model_path}" \
    --model_path="${model_base_dir}/${model}" \
    --output_dir="${output_base_dir}/${model}/qwen_sampling" \
    --data="${data}" \
    --temperature=0.7 \
    --top_p=0.8 \
    --extra_body='{"chat_template_kwargs": {"enable_thinking": false}, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.05}' \
    --storage_mount_dir="${storage_mount_dir}" \
    --storage_ocr_model_path="${storage_ocr_model_path}" \
    --ocr_model_path="${model_base_dir}/${ocr_model}" \
    --tp="${tp}" \
    --dp="${dp}"
```


# Visualization

```bash
streamlit run src/tools/sft_data_visualizer.py --server.port 8443
```
