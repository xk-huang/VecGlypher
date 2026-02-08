"""
python -m src.svg_glyph_gen_v2.render_pangram_for_fonts
"""

# Suppress *all* fontTools warnings
import json
from concurrent.futures import as_completed, ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
from re import escape
from types import SimpleNamespace

import click
import numpy as np
from fontTools import configLogger
from tqdm import tqdm, trange

from .custom_render import renderTextToObj
from .render_text_with_fonttools import text_to_svg
from .utils import load_jsonl, prepare_output_dir_and_logger, write_jsonl

configLogger(level="ERROR")  # or "CRITICAL"


@click.command()
@click.option(
    "--input_metadata_jsonl",
    type=click.Path(exists=True),
    # default="data/processed/google_font_metadata-filter_invalid",
    required=True,
)
@click.option(
    "--input_google_font_dir",
    type=click.Path(),
    # default="data/google_fonts/ofl",
    required=True,
)
@click.option(
    "--output_dir",
    type=click.Path(),
    # default="data/processed/google_font_metadata-filter_invalid-pangram",
    required=True,
)
@click.option("--num_workers", type=int, default=20)
@click.option("--backend", type=str, default="fonttools")
@click.option("--max_samples", default=None, type=int)
@click.option("--batch_size", default=2000, type=int)
@click.option("--only_plot_hist", is_flag=True, default=False)
@click.option("--overwrite", is_flag=True, default=False)
def main(**kargs):
    args = SimpleNamespace(**kargs)

    font_filter = RenderPangram(args)
    if args.only_plot_hist:
        font_filter.plot_hist_svg_len()
        return
    font_filter.render_pangrams()


PANGRAM = "The quick brown fox jumps over the lazy dog"


class RenderPangram:
    def __init__(self, args):
        self.args = args

        # prepare output dir and logger
        should_skip, logger = prepare_output_dir_and_logger(
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        # NOTE: only plot reuse the same output dir
        if should_skip and not args.only_plot_hist:
            exit()
        self.output_dir = Path(args.output_dir)
        self.logger = logger

        self.font_sub_dir = Path(self.args.input_google_font_dir)
        backend = args.backend
        self.output_svg_jsonl = self.output_dir / f"pangram_svg.{backend}.jsonl"

        self.num_workers = args.num_workers

        self.batch_size = args.batch_size

    def render_pangrams(self):
        metadata = []
        input_metadata_jsonl = self.args.input_metadata_jsonl
        input_metadata_jsonl = Path(input_metadata_jsonl)
        if not input_metadata_jsonl.exists():
            raise FileNotFoundError(f"{input_metadata_jsonl} does not exist.")

        metadata = load_jsonl(input_metadata_jsonl, logger=self.logger)
        self.logger.info(f"Loaded {len(metadata)} metadata from {input_metadata_jsonl}")
        if self.args.max_samples:
            metadata = metadata[: self.args.max_samples]
            self.logger.info(f"Limit to {self.args.max_samples} fonts.")

        # test one font
        metadata_item = metadata[0]
        filename = metadata_item["filename"]
        font_family_dir_name = metadata_item["font_family_dir_name"]
        _ = self.get_rendered_svg(filename, font_family_dir_name)

        # load all svg length
        results = []
        batch_size = self.batch_size
        for i in trange(
            0, len(metadata), batch_size, desc=f"Batch processing (size={batch_size})"
        ):
            metadata_chunk = metadata[i : i + batch_size]
            results.extend(self.render_pangram_svg(metadata_chunk))

        # write to jsonl
        write_jsonl(results, self.output_svg_jsonl, logger=self.logger)

    def render_pangram_svg(self, metadata):
        results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for metadata_item in metadata:
                filename = metadata_item["filename"]
                font_family_dir_name = metadata_item["font_family_dir_name"]
                futures.append(
                    executor.submit(
                        self._get_rendered_svg, filename, font_family_dir_name
                    )
                )

            pbar = tqdm(total=len(metadata), desc="Rendering pangram svgs")
            for future in as_completed(futures):
                svg, filename, font_family_dir_name = future.result()
                svg_len = len(svg)

                prompt = f"{svg_len:07d}: {filename}"
                output_dict = {
                    "svg": svg,
                    "svg_len": svg_len,
                    "prompt": prompt,
                    "filename": filename,
                    "font_family_dir_name": font_family_dir_name,
                }
                results.append(output_dict)
                pbar.update(1)

        self.logger.info(f"Done rendering pangram svgs, length {len(results)}")
        return results

    def plot_hist_svg_len(self):
        svg_lens = []
        for i in load_jsonl(self.output_svg_jsonl.parent, logger=self.logger):
            svg_lens.append(i["svg_len"])

        # plot the distribution of svg_len
        self._plot_hist(
            svg_lens, plot_title_suffix=" with quantiles", plot_quantiles=True
        )

        # less than 75%, 90%, 95%
        for quantile in [0.75, 0.9, 0.95]:
            q_svg_lens = [
                svg_len
                for svg_len in svg_lens
                if svg_len < np.quantile(svg_lens, quantile)
            ]
            self._plot_hist(
                q_svg_lens,
                file_name_suffix=f"less_than_{quantile}",
                plot_title_suffix=f" less than {quantile}",
                plot_quantiles=False,
            )

    def _plot_hist(
        self, svg_lens, file_name_suffix="", plot_title_suffix="", plot_quantiles=True
    ):
        import matplotlib.pyplot as plt

        # from matplotlib.ticker import FuncFormatter

        fig, ax = plt.subplots()
        ax.hist(svg_lens, bins=100)

        # Calculate quantiles
        if plot_quantiles:
            quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            q_values = np.quantile(svg_lens, quantiles)

            # Add vertical lines for quantiles
            for q, val in zip(quantiles, q_values):
                q_ = int(q * 100)
                ax.axvline(
                    val,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f"{q_}th percentile: {val:.0f}",
                )

        # Format x-axis to show integers without scientific notation
        ax.ticklabel_format(style="plain", axis="x")
        # OR alternatively:
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))

        ax.legend()
        ax.set_xlabel("SVG Length")
        ax.set_ylabel("Frequency")
        title = "Histogram of SVG Length of Different Fonts"
        if plot_title_suffix:
            title += f"{plot_title_suffix}"
        ax.set_title(title)

        output_hist_dir = self.output_dir / "hist"
        output_hist_dir.mkdir(parents=True, exist_ok=True)
        if file_name_suffix:
            file_name = f"pangram_svg_len_hist.{file_name_suffix}.png"
        else:
            file_name = "pangram_svg_len_hist"
        output_hist_path = output_hist_dir / file_name
        fig.savefig(output_hist_path)
        self.logger.info(f"saved to {output_hist_path}")

    def _get_rendered_svg(self, filename, font_family_dir_name):
        return (
            self.get_rendered_svg(filename, font_family_dir_name),
            filename,
            font_family_dir_name,
        )

    def get_rendered_svg(self, filename, font_family_dir_name):
        font_path = self.font_sub_dir / font_family_dir_name / filename

        backend = self.args.backend
        svg = ""
        try:
            if backend == "fonttools":
                svg = text_to_svg(font_path, PANGRAM)
            elif backend == "blackrenderer":
                buf = BytesIO()
                renderTextToObj(font_path, PANGRAM, buf)
                svg = buf.getvalue().decode("utf-8")
            else:
                raise ValueError(f"Unknown backend: {backend}")
        except Exception as e:
            self.logger.error(f"Error rendering SVG for {font_path}: {e}")

        return svg


if __name__ == "__main__":
    main()
