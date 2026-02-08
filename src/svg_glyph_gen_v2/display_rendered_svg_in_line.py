"""
python -m src.svg_glyph_gen_v2.display_rendered_svg_in_line
--input_svg_dir
--output_dir
"""

# from svgpathtools import svgstr2paths
import json
from pathlib import Path
from types import SimpleNamespace

import click

from .render_normalized_svg import get_new_bounding_box
from .svg_simplifier import SVGSimplifier
from .utils import load_jsonl, prepare_output_dir_and_logger


@click.command()
@click.option(
    "--input_svg_dir",
    type=click.Path(exists=True),
    # default="data/processed/filtered_sft/250903-alphanumeric/ood_font_family/00000.jsonl",
    required=True,
)
@click.option(
    "--output_dir",
    type=click.Path(),
    # default="data/processed/normalized_svg_in_line",
    required=True,
)
@click.option("--target_content_str", type=str, default="a")
@click.option("--num_samples", type=int, default=20)
@click.option("--final_aescender", type=float, default=1000)
@click.option("--width", type=float, default=700)
@click.option("--overwrite", is_flag=True, default=False)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    # prepare output dir and logger
    should_skip, logger = prepare_output_dir_and_logger(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    if should_skip:
        exit()
    output_dir = Path(args.output_dir)
    logger = logger

    svg_simplifer = SVGSimplifier()
    data = load_jsonl(args.input_svg_dir)

    target_content_str = args.target_content_str
    logger.info(f"target_content_str: {target_content_str}")
    data = filter_by_constent_str(data, target_content_str)

    final_aescender = args.final_aescender
    logger.info(f"final_aescender: {final_aescender}")
    width = args.width
    logger.info(f"width: {width}")

    output_svg = ""
    num_svg = 0
    for i, d in enumerate(data):
        if i >= args.num_samples:
            break
        encoded_svg = d["output"]
        logger.info(f"metadata [{i}]: {d['metadata']}")

        offset_transform = f"transform='translate({num_svg * width}, 0)'"
        encoded_svg = encoded_svg.replace("/>", f" {offset_transform}/>")
        encoded_svg += f"""<path d="M {num_svg * width} 0 L {num_svg * width} {final_aescender}" stroke="black" stroke-width="3" fill="none"/>\n"""

        output_svg += encoded_svg
        num_svg += 1

    # add a new line y=final_aescender
    output_svg += f"""<path d="M 0 {final_aescender} L {width * num_svg} {final_aescender}" stroke="black" stroke-width="3" fill="none"/>\n"""

    decoded_svg = svg_simplifer.decode(output_svg)

    output_svg_path = output_dir / f"svg_in_line-{target_content_str}.svg"
    output_svg_path.write_text(decoded_svg)
    logger.info(f"save svg to: {output_svg_path}")


def filter_by_constent_str(data, target_content_str):
    filtered_data = []
    for d in data:
        metadata = json.loads(d["metadata"])
        if metadata["content_str"] == target_content_str:
            filtered_data.append(d)
    return filtered_data


if __name__ == "__main__":
    main()
