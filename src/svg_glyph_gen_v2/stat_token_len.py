"""
python -m src.svg_glyph_gen_v2.stat_token_len
"""

import json
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from .build_sft_data_v2 import WORD_SEP
from .utils import load_jsonl, prepare_output_dir_and_logger


def stat_token(data, tokenizer):
    instruction = data["instruction"]
    system = data["system"]
    output = data["output"]

    input_str = f"{instruction} {system}"
    output_str = output

    input_token = tokenizer(input_str, add_special_tokens=True)["input_ids"]
    output_token = tokenizer(output_str, add_special_tokens=True)["input_ids"]

    input_token_len = len(input_token)
    output_token_len = len(output_token)

    metadata = json.loads(data["metadata"])
    content_str = metadata["content_str"].replace(WORD_SEP, "")
    return {
        "input_str_len": len(input_str),
        "input_token_len": input_token_len,
        "output_str_len": len(output_str),
        "output_token_len": output_token_len,
        "content_len": len(content_str),
    }


def batch_stat_token(data_list, tokenizer):
    token_len_list = []
    for data in data_list:
        stat = stat_token(data, tokenizer)
        token_len_list.append(stat)
    return token_len_list


def stat_token_from_data_list(data_list, tokenizer, num_workers, batch_size):
    token_len_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future2data = []
        for start_idx in range(0, len(data_list), batch_size):
            end_idx = min(start_idx + batch_size, len(data_list))
            data_batch = data_list[start_idx:end_idx]
            future = executor.submit(batch_stat_token, data_batch, tokenizer)
            future2data.append(future)

        for future in tqdm(as_completed(future2data), total=len(future2data)):
            try:
                stat = future.result()
                token_len_list.extend(stat)
            except Exception as exc:
                raise exc
    return token_len_list


def plot_hist(token_len_df, data_name, output_dir, skip_scatter=False, logger=None):
    fig, ax = plt.subplots(figsize=(12, 7), nrows=2, ncols=3)
    ax[0][0].hist(token_len_df["input_str_len"], bins=100)
    ax[0][0].set_title("input_str_len", fontdict={"fontsize": 10})
    ax[0][1].hist(token_len_df["input_token_len"], bins=100)
    ax[0][1].set_title("input_token_len", fontdict={"fontsize": 10})
    # Add quantile lines
    quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    quantile_values = token_len_df["input_token_len"].quantile(quantiles)
    for q, val in zip(quantiles, quantile_values):
        ax[0][1].axvline(
            x=val,
            color="r",
            linestyle="--",
            alpha=0.7,
            linewidth=0.8,
            label=f"{int(q*100)}% quantile: {int(val)}",
        )
    ax[0][1].legend(fontsize=8)
    # scatter plot of content_len vs output_token_len
    y_max = max(ax[0][0].get_ylim()[-1], ax[0][1].get_ylim()[1])
    x_max = max(ax[0][0].get_xlim()[-1], ax[0][1].get_xlim()[1])
    for i in range(2):
        ax[0][i].set_ylim(0, y_max)
        ax[0][i].set_xlim(0, x_max)

    ax[1][0].hist(token_len_df["output_str_len"], bins=100)
    ax[1][0].set_title("output_str_len", fontdict={"fontsize": 10})
    ax[1][1].hist(token_len_df["output_token_len"], bins=100)
    ax[1][1].set_title("output_token_len", fontdict={"fontsize": 10})
    # Add quantile lines
    quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    quantile_values = token_len_df["output_token_len"].quantile(quantiles)
    for q, val in zip(quantiles, quantile_values):
        ax[1][1].axvline(
            x=val,
            color="r",
            linestyle="--",
            alpha=0.7,
            linewidth=0.8,
            label=f"{int(q*100)}% quantile: {int(val)}",
        )
    ax[1][1].legend(fontsize=8)

    # hist of content_len
    ax[0][2].hist(token_len_df["content_len"], bins=100)
    ax[0][2].set_title("content_len hist", fontdict={"fontsize": 10})
    ax[0][2].set_xlabel("content_len")
    ax[0][2].set_ylabel("frequency")

    y_max = max(ax[1][0].get_ylim()[-1], ax[1][1].get_ylim()[1])
    x_max = max(ax[1][0].get_xlim()[-1], ax[1][1].get_xlim()[1])
    for i in range(2):
        ax[1][i].set_ylim(0, y_max)
        ax[1][i].set_xlim(0, x_max)
    for ax_ in ax.flatten():
        ax_.set_xlabel("length", fontsize=8)
        ax_.set_ylabel("count", fontsize=8)
        ax_.tick_params(axis="both", which="major", labelsize=8)
    # scatter plot of content_len vs output_token_len
    if not skip_scatter:
        ax[1][2].scatter(
            token_len_df["content_len"], token_len_df["output_token_len"], s=0.5
        )
        ax[1][2].set_xlabel("content_len")
        ax[1][2].set_ylabel("output_token_len")
        ax[1][2].set_title("content_len vs output_token_len", fontdict={"fontsize": 10})
        ax[1][2].set_xlim(0, ax[1][2].get_xlim()[-1])
        ax[1][2].set_ylim(0, ax[1][2].get_ylim()[-1])

    fig_title = f"token/string length distribution {data_name}"
    fig.suptitle(fig_title)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{data_name}.pdf"
    if logger is not None:
        logger.info(f"saving to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if logger is not None:
        logger.info(f"save to {output_path}")


DEFAULT_DATASET_DIR = "data/processed/filtered_sft/250903-alphanumeric"
DEFAULT_DATASET_NAME = "ood_font_family"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR
DEFAULT_MODEL = (
    "/home/vecglypher/mnt/workspace/hf_downloads/Qwen/Qwen3-4B"
)


@click.command()
@click.option(
    "--dataset_dir",
    type=click.Path(),
    # default=DEFAULT_DATASET_DIR,
    required=True,
)
@click.option(
    "--dataset_name",
    type=click.Path(),
    # default=DEFAULT_DATASET_NAME,
    required=True,
)
@click.option(
    "--output_dir",
    type=click.Path(),
    # default=DEFAULT_OUTPUT_DIR,
    required=True,
)
@click.option(
    "--model_name_or_path",
    type=click.Path(),
    default=DEFAULT_MODEL,
)
@click.option("--skip_scatter/--no_skip_scatter", default=False)
@click.option("--num_workers", type=int, default=20)
@click.option("--batch_size", type=int, default=3000)
@click.option("--skip_plot/--no_skip_plot", default=False)
@click.option("--overwrite", is_flag=True, default=False)
def main(
    dataset_dir,
    dataset_name,
    output_dir,
    model_name_or_path,
    skip_scatter,
    num_workers,
    batch_size,
    overwrite,
    skip_plot,
):
    should_skip, logger = prepare_output_dir_and_logger(
        output_dir=output_dir,
        overwrite=overwrite,
    )
    if should_skip:
        exit()
    output_dir = Path(output_dir)
    output_df_path = output_dir / f"{dataset_name}.tsv"
    logger.info(f"working on {output_dir} for {dataset_name}")

    if output_df_path.exists() and not overwrite:
        logger.info(f"output_df_path {output_df_path} exists, skip")
        return
    else:

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        additional_special_tokens = [WORD_SEP]
        tokenizer.add_special_tokens(
            {"additional_special_tokens": additional_special_tokens},
            replace_additional_special_tokens=False,
        )
        logger.info(f"tokenizer.special_tokens_map: {tokenizer.special_tokens_map}")

        dataset_dir = Path(dataset_dir)
        jsonl_dir = dataset_dir / dataset_name
        logger.info(f"jsonl_dir: {jsonl_dir}")
        data_list = load_jsonl(jsonl_dir, logger=logger)

        token_len_list = stat_token_from_data_list(
            data_list, tokenizer, num_workers, batch_size
        )

        token_len_df = pd.DataFrame(token_len_list)
        token_len_df.to_csv(output_df_path, sep="\t", index=False)

    if not skip_plot:
        plot_title = f"{dataset_dir.name}-{dataset_name}"
        logger.info(f"plotting {plot_title}")
        plot_hist(token_len_df, plot_title, output_dir, skip_scatter)


if __name__ == "__main__":
    main()
