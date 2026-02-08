#!/usr/bin/env python3
"""
Process JSONL files under a directory (recursively) and write results to score.jsonl.

Usage:
    python -m src.eval.score_img_eval \
        <input_jsonl_dir> \
        <input_predict_img_base64_jsonl_dir> \
        <input_label_img_base64_jsonl_dir> \
        [<output_dir>]

- Finds all *.jsonl under input_dir (including subdirectories).
- Streams records line-by-line, so it works on large files.
- Applies `process_record(record)` to each parsed dict.
- Writes all processed records to a single file: <output_dir>/score.jsonl

Run:
    output_dir=<output_dir>
    python -m src.serve.decode_to_svg ${output_dir}/infer
    python -m src.eval.svg2img_dir "${output_dir}"/infer_decoded "${output_dir}"/infer_decoded-img_base64-predict --field predict --width 192 --height 192
    python -m src.eval.svg2img_dir "${output_dir}"/infer_decoded "${output_dir}"/infer_decoded-img_base64-label --field label --width 192 --height 192

    HF_DINO_MODEL_PATH=${storage_base_dir}/workspace/hf_downloads/facebook/dinov2-base \
    HF_CLIP_MODEL_PATH=${storage_base_dir}/workspace/hf_downloads/openai/clip-vit-base-patch32 \
    TORCH_HUB_CKPT_DIR=${storage_base_dir}/workspace/hf_downloads/eval_ckpts \
    OPENAI_CLIP_CACHE_DIR=${storage_base_dir}/workspace/hf_downloads/eval_ckpts \
    CUDA_VISIBLE_DEVICES=0 \
    python -m src.eval.score_img_eval \
        "${output_dir}"/infer_decoded  \
        "${output_dir}"/infer_decoded-img_base64-predict \
        "${output_dir}"/infer_decoded-img_base64-label \
        "${output_dir}"/results_img_eval \
        --metrics CLIPScore

for all metrics, check src/eval/metrics/metrics.py
"""


import base64
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import click
import pandas as pd
import tqdm
from PIL import Image

from ..svg_glyph_gen_v2.build_sft_data_v2 import WORD_SEP
from ..svg_glyph_gen_v2.filter_by_pangram_svg import blake2_hash
from ..svg_glyph_gen_v2.utils import (
    count_lines,
    load_jsonl_by_generator,
    prepare_output_dir_and_logger,
)
from .metrics.metrics import SVGMetrics

SCORE_FILE_NAME = "score.jsonl"


processed_record_cnt = 0


def load_base64_to_img(b64_str):
    img_data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_data))
    return img


def process_record(record: Dict[str, Any], predict, label) -> Dict[str, Any]:
    """
    Dummy transformation. Modify this function to implement your logic.

    Return:
        - A dict to write to the output JSONL
        - None to skip writing this record
    """
    global processed_record_cnt

    # Example: pass through and add a dummy score
    gen_svg_text = record["predict"]
    gen_img_hash = blake2_hash(gen_svg_text)
    gen_img_hash_2 = predict["img_hash"]
    if gen_img_hash != gen_img_hash_2:
        raise ValueError(f"img_hash {gen_img_hash} != img_hash_2 {gen_img_hash_2}")
    gen_svg_img = load_base64_to_img(predict["img_base64"])

    gt_svg_text = record["label"]
    gt_img_hash = blake2_hash(gt_svg_text)
    gt_img_hash_2 = label["img_hash"]
    if gt_img_hash != gt_img_hash_2:
        raise ValueError(f"img_hash {gt_img_hash} != img_hash_2 {gt_img_hash_2}")
    gt_svg_img = load_base64_to_img(label["img_base64"])

    json_dict = {
        "sample_id": processed_record_cnt,
        "metadata": record["metadata"],
    }
    processed_record_cnt += 1

    metadata_str = record["metadata"]
    metadata = json.loads(metadata_str)
    # content_str = metadata["content_str"]
    # content_str = content_str.replace(WORD_SEP, "")
    # replace word separator "<|SEP|>"
    sft_instruction = metadata["sft_instruction"]
    sft_instruction = sft_instruction.replace(WORD_SEP, "")

    # check img size
    if gen_svg_img.size != gt_svg_img.size:
        raise ValueError(
            f"gen_svg_img size {gen_svg_img.size} != gt_svg_img size {gt_svg_img.size}"
        )

    return {
        "gen_svg": gen_svg_text,
        "gt_svg": gt_svg_text,
        "gen_im": gen_svg_img,
        "gt_im": gt_svg_img,
        "json": json_dict,
        "caption": sft_instruction,
    }


def iter_jsonl_files(root: Path) -> Iterable[Path]:
    """Yield all *.jsonl files under root recursively."""
    for i in root.rglob("*.jsonl"):
        if i.is_file() and i.name != SCORE_FILE_NAME:
            yield i


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument(
    "input_jsonl_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument(
    "input_predict_img_base64_jsonl_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument(
    "input_label_img_base64_jsonl_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--max_samples", type=int, default=None, help="Max number of samples to process."
)
@click.option("--metrics", type=str, default=None, help="sep by comma, default is all")
@click.option("--overwrite", is_flag=True, default=False)
def main(
    input_jsonl_dir: Path,
    input_predict_img_base64_jsonl_dir: Path,
    input_label_img_base64_jsonl_dir: Path,
    output_dir: Path,
    max_samples: Optional[int],
    metrics: Optional[str],
    overwrite: bool,
) -> None:
    """
    INPUT_DIR: Directory to search for *.jsonl (recursively).
    OUTPUT_DIR: Optional output directory. If not provided, uses INPUT_DIR.
    """
    should_skip, logger = prepare_output_dir_and_logger(
        output_dir=output_dir,
        overwrite=overwrite,
    )
    if should_skip:
        exit()

    num_samples_0 = count_lines(input_jsonl_dir)
    num_samples_1 = count_lines(input_predict_img_base64_jsonl_dir)
    num_samples_2 = count_lines(input_label_img_base64_jsonl_dir)
    if num_samples_0 != num_samples_1 or num_samples_0 != num_samples_2:
        raise ValueError(
            f"lengths are not equal:\n- input_jsonl_dir: {num_samples_0}\n- input_predict_img_base64_jsonl_dir: {num_samples_1}\n- input_label_img_base64_jsonl_dir: {num_samples_2}"
        )

    if metrics is not None:
        metrics_config = {}
        for m in metrics.split(","):
            metrics_config[m.strip()] = True
    else:
        metrics_config = None
    logger.info(f"metrics_config: {metrics_config}")
    svg_metrics = SVGMetrics(metrics_config)

    batch = defaultdict(list)

    cnt = 0
    max_samples = num_samples_0 if max_samples is None else max_samples
    jsonl_iter = load_jsonl_by_generator(input_jsonl_dir)
    predict_iter = load_jsonl_by_generator(input_predict_img_base64_jsonl_dir)
    label_iter = load_jsonl_by_generator(input_label_img_base64_jsonl_dir)

    pbar = tqdm.tqdm(total=max_samples)
    for record, predict, label in zip(jsonl_iter, predict_iter, label_iter):
        if cnt >= max_samples:
            break

        processed = process_record(record, predict, label)
        for k, v in processed.items():
            batch[k].append(v)
        cnt += 1
    logger.info(f"Processed {cnt} records")

    batch = dict(batch)
    avg_results_dict, all_results_dict = svg_metrics.calculate_metrics(batch)

    logger.info(f"Writing results to: {output_dir}")
    avg_results_dict["num_samples"] = cnt
    # Save average results
    json_path = output_dir / "results_avg.json"
    with open(json_path, "w") as f:
        json.dump(avg_results_dict, f, indent=4, sort_keys=True)
    logger.info(f"Saved average results to: {json_path}")

    # Save detailed results
    df = pd.DataFrame.from_dict(all_results_dict, orient="index")
    df_path = output_dir / "all_results.csv"
    df.to_csv(df_path)
    logger.info(f"Saved detailed results to: {df_path}")


if __name__ == "__main__":
    main()
