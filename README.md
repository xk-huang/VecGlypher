# VecGlypher

[![Project Page Link](https://img.shields.io/badge/Project-Page-2ea44f?logo=googlechrome&logoColor=white)](https://xk-huang.github.io/VecGlypher)
[![Paper Link](https://img.shields.io/badge/Paper-VecGlypher-1f6feb?logo=readthedocs&logoColor=white)](https://arxiv.org/abs/2602.21461)
[![Hugging Face Profile](https://img.shields.io/badge/huggingface-VecGlypher-orange?logo=huggingface)](https://huggingface.co/VecGlypher)
[![Venue](https://img.shields.io/badge/Venue-CVPR%202026-red)](https://xk-huang.github.io/VecGlypher)
[![Date](https://img.shields.io/badge/Date-2026--02--25-9c27b0)](https://xk-huang.github.io/VecGlypher)

This repository contains the re-implementation for **VecGlypher: Unified Vector Glyph Generation with Language Models**.

VecGlypher formulates glyph generation as language modeling over SVG path tokens. A single multimodal LLM supports:
- text-referenced glyph generation (style tags/text + target character),
- image-referenced glyph generation (reference glyph images + target character),
- direct one-pass SVG output without raster-to-vector post-processing.

![Teaser from VecGlypher paper](docs/assets/figure_1.png)

![Method from VecGlypher paper](docs/assets/figure_3.png)


Key insight on current LLM limits:
- Many general LLMs that can generate generic SVG graphics (for example icons or simple drawings) still struggle to draw valid glyphs and to follow typography style instructions.
- A likely reason is data availability: glyph programs are rarely seen in pretraining corpora because most font data is stored in binary files such as `.ttf` and `.otf`, not as explicit SVG path programs.
- This suggests a practical generalization boundary: strong general SVG/code ability does not automatically transfer to high-fidelity typographic glyph generation without targeted data and training.


## Quick Start

- [environment.md](docs/public/environment.md) for setup and installation.
- [data.md](docs/public/data.md) for dataset building and processing.
- [training.md](docs/public/training.md) for training new models or continuing training.
- [inference.md](docs/public/inference.md) for running inference with trained models.
- [evaluation.md](docs/public/evaluation.md) for decoding and evaluation tools.
- [troubleshooting.md](docs/public/troubleshooting.md) for common issues and fixes.

<details><summary>Repository Map.</summary>

- `docs/public/`: user-facing setup/training/inference/eval docs.
- `src/svg_glyph_gen_v2/`: font/glyph processing and dataset construction.
- `src/sft/`: training entry points and config files.
- `src/serve/`: inference clients, decoding, and serving helpers.
- `src/eval/`: OCR/image/point-cloud style evaluation utilities.
- `scripts/data_process/`: end-to-end dataset build scripts.
- `scripts/eval_locally/`: evaluation workflows.

</details>


## Citation

If you use this codebase, please cite the VecGlypher paper from the project page above.

```
@inproceedings{huang2026vecglypher,
   title = {VecGlypher: Unified Vector Glyph Generation with Language Models},
   author = {Xiaoke Huang, Bhavul Gauri, Kam Woh Ng, Tony Ng, Mengmeng Xu, Zhiheng Liu, Weiming Ren, Zhaochong An, Zijian Zhou, Haonan Qiu, Yuyin Zhou, Sen He, Ziheng Wang, Tao Xiang, Xiao Han},
   booktitle = {CVPR},
   year = {2026},
}
```
