"""
sft_base_dir=data/processed/sft/250903-alphanumeric/
split_names=(
  "train_font_family"
  "ood_font_family"
)
for split_name in ${split_names[@]}; do
TOKENIZERS_PARALLELISM=true python -m src.tools.stat_token_len_for_paper \
--dataset_dir ${sft_base_dir} \
--dataset_name ${split_name} \
--output_dir misc/stat_token_len_for_paper/google_fonts-sft/${split_name}
done


sft_base_dir=data/processed_envato/sft/250903-envato-alphanumeric/
split_names=(
train_font_family
ood_font_family
)
for split_name in ${split_names[@]}; do
TOKENIZERS_PARALLELISM=true python -m src.tools.stat_token_len_for_paper \
  --dataset_dir ${sft_base_dir} \
  --dataset_name ${split_name} \
  --output_dir misc/stat_token_len_for_paper/envato_fonts-sft/${split_name}
done


sft_base_dir=data/processed/filtered_sft/250903-alphanumeric/
split_names=(
  "train_font_family"
  "ood_font_family_decon"
)
for split_name in ${split_names[@]}; do
TOKENIZERS_PARALLELISM=true python -m src.tools.stat_token_len_for_paper \
--dataset_dir ${sft_base_dir} \
--dataset_name ${split_name} \
--output_dir misc/stat_token_len_for_paper/google_fonts-filtered_sft/${split_name}
done


sft_base_dir=data/processed_envato/filtered_sft/250903-envato-alphanumeric
split_names=(
train_font_family
ood_font_family
)
for split_name in ${split_names[@]}; do
TOKENIZERS_PARALLELISM=true python -m src.tools.stat_token_len_for_paper \
  --dataset_dir ${sft_base_dir} \
  --dataset_name ${split_name} \
  --output_dir misc/stat_token_len_for_paper/envato_fonts-filtered_sft/${split_name}
done


tar -czf misc/stat_token_len_for_paper.tar.gz $(find misc/stat_token_len_for_paper -type f -name "*.pdf")
"""

import json
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer

from ..svg_glyph_gen_v2.build_sft_data_v2 import WORD_SEP
from ..svg_glyph_gen_v2.utils import load_jsonl, prepare_output_dir_and_logger


plt.rcParams.update(
    {
        "font.family": "serif",
    }
)
sns.set_style("ticks")
sns.set_context(
    "paper",
    font_scale=2,
)
sns.set_palette("pastel")


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


def plot_hist(
    token_len_df,
    data_name,
    output_dir,
    logger=None,
    plot_quantile=False,
    figsize=(8, 6),
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # input token len
    fig, ax = plt.subplots(figsize=figsize)
    # ax.hist(
    #     token_len_df["input_token_len"],
    #     bins=20,
    #     color="steelblue",
    #     alpha=0.7,
    # )
    sns.histplot(
        token_len_df["input_token_len"].values,
        bins=20,
        kde=False,
        ax=ax,
        stat="probability",
    )
    sns.despine()

    # Get current ticks
    # x_ticks = ax.get_xticks()
    # y_ticks = ax.get_yticks()

    # # Make both axes have the same number of ticks
    # num_ticks = min(len(x_ticks), len(y_ticks))
    # ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num_ticks))
    # ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num_ticks))

    # ax.set_title("input_token_len", fontdict={"fontsize": 10})
    ax.set_xlabel("Input Token Length")
    ax.set_ylabel("Probability")
    # ax.tick_params(
    #     axis="both",
    #     which="major",
    # )
    # Add quantile lines
    if plot_quantile:
        quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        quantile_values = token_len_df["input_token_len"].quantile(quantiles)
        for q, val in zip(quantiles, quantile_values):
            ax.axvline(
                x=val,
                color="r",
                linestyle="--",
                label=f"{int(q*100)}% quantile: {int(val)}",
            )
        ax.legend(
            # fontsize=8,
        )

    output_path = output_dir / f"{data_name}-input_token_len.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if logger is not None:
        logger.info(f"save to {output_path}")

    # output token len
    fig, ax = plt.subplots(figsize=figsize)
    # ax.hist(
    #     token_len_df["output_token_len"],
    #     bins=50,
    #     color="steelblue",
    #     alpha=0.7,
    # )
    sns.histplot(
        token_len_df["output_token_len"],
        bins=50,
        kde=False,
        ax=ax,
        stat="probability",
    )
    sns.despine()

    ax.set_xlabel("Output Token Length")
    ax.set_ylabel("Probability")
    # ax.tick_params(
    #     axis="both",
    #     which="major",
    # )
    # Add quantile lines
    if plot_quantile:
        quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        quantile_values = token_len_df["output_token_len"].quantile(quantiles)
        for q, val in zip(quantiles, quantile_values):
            ax.axvline(
                x=val,
                color="r",
                linestyle="--",
                label=f"{int(q*100)}% quantile: {int(val)}",
            )
        ax.legend(
            # fontsize=8,
        )

    output_path = output_dir / f"{data_name}-output_token_len.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
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
@click.option("--num_workers", type=int, default=20)
@click.option("--batch_size", type=int, default=3000)
@click.option("--skip_plot/--no_skip_plot", default=False)
@click.option("--overwrite", is_flag=True, default=False)
def main(
    dataset_dir,
    dataset_name,
    output_dir,
    model_name_or_path,
    num_workers,
    batch_size,
    overwrite,
    skip_plot,
):
    should_skip, logger = prepare_output_dir_and_logger(
        output_dir=output_dir,
        overwrite=overwrite,
    )

    output_dir = Path(output_dir)
    output_df_path = output_dir / f"{dataset_name}.tsv"
    logger.info(f"working on {output_dir} for {dataset_name}")

    dataset_dir = Path(dataset_dir)

    output_df_pkl_path = output_dir / f"{dataset_name}.pkl"
    if output_df_pkl_path.exists():
        logger.info(f"loading from {output_df_pkl_path}")
        token_len_df = pd.read_pickle(output_df_pkl_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        additional_special_tokens = [WORD_SEP]
        tokenizer.add_special_tokens(
            {"additional_special_tokens": additional_special_tokens},
            replace_additional_special_tokens=False,
        )
        logger.info(f"tokenizer.special_tokens_map: {tokenizer.special_tokens_map}")

        jsonl_dir = dataset_dir / dataset_name
        logger.info(f"jsonl_dir: {jsonl_dir}")
        data_list = load_jsonl(jsonl_dir, logger=logger)

        token_len_list = stat_token_from_data_list(
            data_list, tokenizer, num_workers, batch_size
        )

        token_len_df = pd.DataFrame(token_len_list)
        token_len_df.to_csv(output_df_path, sep="\t", index=False)

        token_len_df.to_pickle(output_df_pkl_path)

    # dump stat dict
    num_samples = len(token_len_df)
    total_input_token_len = token_len_df["input_token_len"].sum()
    total_output_token_len = token_len_df["output_token_len"].sum()
    avg_input_token_len = total_input_token_len / num_samples
    avg_output_token_len = total_output_token_len / num_samples
    stat_dict = {
        "num_samples": num_samples,
        "total_input_token_len": int(total_input_token_len),
        "total_output_token_len": int(total_output_token_len),
        "avg_input_token_len": float(avg_input_token_len),
        "avg_output_token_len": float(avg_output_token_len),
    }
    stat_dict_output_path = output_dir / "stat_dict.json"
    with open(stat_dict_output_path, "w") as f:
        json.dump(stat_dict, f)
    logger.info(f"dump stat dict to: {stat_dict_output_path}")

    if not skip_plot:
        plot_title = f"{dataset_dir.name}-{dataset_name}"
        logger.info(f"plotting {plot_title}")
        plot_hist(token_len_df, plot_title, output_dir, logger)


if __name__ == "__main__":
    main()
