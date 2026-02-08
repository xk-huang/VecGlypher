# Troubleshooting

- Missing `fontTools`: `pip install fonttools`
- Cairo/Skia errors: install system libraries for cairo and freetype
- GPU errors: verify CUDA driver compatibility with your PyTorch build
- OOM: reduce batch size, sequence length, or number of workers
- vLLM connection issues: confirm `--port` and `--base_url` match
