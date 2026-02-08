"""
python -m src.tools.parse_lmm_ocr_score [input_file] [output_file]
input_files=(
data/processed_envato/filter_fonts_by_lmm_ocr/results_ocr_eval-Qwen2.5-VL-7B-Instruct-acc-use_case/score.jsonl
data/processed_envato/filter_fonts_by_lmm_ocr/results_ocr_eval-Qwen2.5-VL-32B-Instruct-acc-use_case/score.jsonl
data/processed_envato/filter_fonts_by_lmm_ocr/results_ocr_eval-Qwen3-VL-30B-A3B-Instruct-acc-use_case/score.jsonl
)
for input_file in ${input_files[@]}; do
    output_name=$(basename $(dirname $input_file))
    python -m src.tools.parse_lmm_ocr_score ${input_file} misc/debug-lmm_ocr_score/${output_name}
done
"""

import logging
from os import error
from pathlib import Path

import click
import pandas as pd

from tqdm import tqdm

from ..svg_glyph_gen_v2.utils import load_jsonl


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("input_file", type=click.Path(exists=True), required=True)
@click.argument("output_dir", type=click.Path(), required=True)
def main(input_file, output_dir):
    logger.info(f"input_file: {input_file}")
    KEYS = [
        "parsed_predict",
        "predict",
        "score",
        "parsed_gt",
        "label",
    ]
    parsed_dict_list = []
    for line_dict in tqdm(load_jsonl(input_file)):
        line_dict: dict
        parsed_dict = {}
        for key in KEYS:
            parsed_dict[key] = line_dict[key]
        parsed_dict_list.append(parsed_dict)

    df = pd.DataFrame(parsed_dict_list)
    stat_df(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "parsed_df.tsv"
    logger.info(f"output_file: {output_file}")
    df.to_csv(output_file, sep="\t", index=False)

    score_false_df = df[df["score"] == False]
    output_file = output_dir / "score_false_df.tsv"
    logger.info(f"output_file: {output_file}")
    score_false_df.to_csv(output_file, sep="\t", index=False)

    # try to remove whitespace in predict and re-calculate accuracy
    parsed_predict = df["parsed_predict"]
    parsed_gt = df["parsed_gt"]
    acc = (parsed_predict == parsed_gt).sum() / len(parsed_predict)

    remove_whitespace_predict = parsed_predict.apply(remove_whitespace)
    remove_acc = (remove_whitespace_predict == parsed_gt).sum() / len(parsed_predict)
    logger.info(f"acc: {acc}, remove_acc: {remove_acc}")


def remove_whitespace(s):
    return "".join(s.split())


def stat_df(df):
    total = len(df)
    score_false = (df["score"] == False).sum()
    score_true = total - score_false
    accuracy = score_true / total
    error_rate = 1 - accuracy
    logger.info(
        f"total: {total}, score_false: {score_false} ({error_rate}), score_true: {score_true} ({accuracy})"
    )


if __name__ == "__main__":
    main()
