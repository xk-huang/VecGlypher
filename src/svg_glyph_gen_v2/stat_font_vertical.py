"""
python -m src.svg_glyph_gen_v2.stat_font_vertical
--input_gfont_metadata
--output_dir
--input_google_font_dir
"""

import math
import shutil
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path

import click
import IPython
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from fontTools.ttLib import TTFont

from .utils import load_jsonl, prepare_output_dir_and_logger


@click.command()
@click.option(
    "--input_gfont_metadata",
    type=click.Path(exists=True),
    # default="data/processed/google_font_metadata-filter_invalid-filter_by_pangram/",
    required=True,
)
@click.option(
    "--output_dir",
    type=str,
    # default="data/processed/google_font_metadata-stat_font_vertical",
    required=True,
)
@click.option(
    "--input_google_font_dir",
    type=str,
    # default="data/google_fonts/ofl",
    required=True,
)
@click.option("--num_workers", type=int, default=20)
@click.option("--font_scale", type=float, default=1000)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
def main(
    input_gfont_metadata,
    output_dir,
    input_google_font_dir,
    overwrite,
    num_workers,
    debug,
    font_scale,
):
    # prepare output dir and logger
    should_skip, logger = prepare_output_dir_and_logger(
        output_dir=output_dir,
        overwrite=overwrite,
    )
    if should_skip:
        exit()
    output_dir = Path(output_dir)
    logger = logger

    data = load_jsonl(input_gfont_metadata, logger=logger)
    font_path_list = [
        build_font_path(metdata_item, input_google_font_dir) for metdata_item in data
    ]

    # test load one sample
    _ = inspect_font_vertical(font_path_list[0])

    output_tsv_path = output_dir / "font_vertical.tsv"
    if output_tsv_path.exists() and not overwrite:
        logger.info(f"Output file already exists: {output_tsv_path}")
        df = pd.read_csv(output_tsv_path, sep="\t")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for font_path in font_path_list:
                futures.append(executor.submit(inspect_font_vertical, font_path))

        results = []
        for future in as_completed(tqdm.tqdm(futures)):
            results.append(future.result())

        df = pd.DataFrame(results)
        df.to_csv(output_tsv_path, sep="\t", index=False)
        logger.info(f"Font vertical info saved to: {output_tsv_path}")

    # plot normalized histogram, by upm=font_scale
    normalized_df = df.copy()
    original_upm = normalized_df["upm"].copy()
    for key in normalized_df.columns:
        if key == "font_path":
            continue
        normalized_df[key] = normalized_df[key] / original_upm * font_scale

    # get quantile 0.5 for each key
    normalized_df_without_str = normalized_df.drop(columns=["font_path"])
    quantile_0_5_df = normalized_df_without_str.quantile(0.5)
    mean_df = normalized_df_without_str.mean()
    mean_quantile_0_5_df = pd.concat(
        [quantile_0_5_df, mean_df], axis=1, keys=["quantile_0_5", "mean"]
    )
    mean_quantile_0_5_df_output_path = (
        output_dir / "font_vertical-mean_quantile_0_5.tsv"
    )
    mean_quantile_0_5_df.to_csv(mean_quantile_0_5_df_output_path, sep="\t", index=True)
    logger.info(f"Font vertical info saved to: {mean_quantile_0_5_df_output_path}")

    plot_hist(df, output_dir, "font_vertical.png", logger)
    plot_hist(normalized_df, output_dir, "font_vertical-normalized.png", logger)

    if debug:
        IPython.embed()


def plot_hist(df, output_dir, output_file_name, logger):
    # plot histogram
    keys = df.columns.tolist()
    # remove "font_path"
    keys.remove("font_path")

    # plot two rows
    num_rows = 2
    num_cols = math.ceil(len(keys) / num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 4))

    for i, key in enumerate(keys):
        ax = axes[i // num_cols, i % num_cols]
        df[key].hist(ax=ax, bins=100)
        ax.set_title(key)
        ax.set_xlabel("value")
        ax.set_ylabel("frequency")
        # plot quantile vertical lines, 25, 50, 75
        quantiles = df[key].quantile([0.0, 0.25, 0.5, 0.75, 1.0])
        for q, val in quantiles.items():
            ax.axvline(
                val, color="r", linestyle="--", label=f"{int(q*100)}th ({val}) pct"
            )
        # Only add legend once per subplot
        ax.legend()

    fig.tight_layout()
    output_fig_path = output_dir / output_file_name
    fig.savefig(output_fig_path)
    logger.info(f"Figure saved to: {output_fig_path}")


def build_font_path(metdata_item, input_google_font_dir):
    font_family_dir_name = metdata_item["font_family_dir_name"]
    filename = metdata_item["filename"]
    font_path = Path(input_google_font_dir) / font_family_dir_name / filename
    return font_path


def inspect_font_vertical(font_path):
    font = TTFont(font_path)
    hhea = font["hhea"]

    ascender = hhea.ascent
    descender = hhea.descent
    line_gap = hhea.lineGap

    try:
        os2 = font["OS/2"]
    except Exception:
        os2 = None

    typo_ascender = None
    typo_descender = None
    typo_linegap = None
    if os2 is not None:
        typo_ascender = os2.sTypoAscender
        typo_descender = os2.sTypoDescender
        typo_linegap = os2.sTypoLineGap

    upm = font["head"].unitsPerEm

    return {
        "font_path": str(font_path),
        "ascender": ascender,
        "descender": descender,
        "line_gap": line_gap,
        "upm": upm,
        "typo_ascender": typo_ascender,
        "typo_descender": typo_descender,
        "typo_linegap": typo_linegap,
    }


if __name__ == "__main__":
    main()
