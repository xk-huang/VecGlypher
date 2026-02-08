"""
python -m src.svg_glyph_gen_v2.svg_simplifier both '<svg xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink" baseProfile="full" height="100%" version="1.1" viewBox="0 0 10 10" width="100%"><defs/><path d="m0 0l0 10l 10 0" transform="translate(0,0)"/></svg>'
python -m src.svg_glyph_gen_v2.svg_simplifier decode ''
python -m src.svg_glyph_gen_v2.svg_simplifier decode
"""

import re
import sys

import click
from svgpathtools import svgstr2paths

from .render_normalized_svg import get_new_bounding_box
from .svg_cleaner import (
    apply_g_attributes_to_children,
    remove_defs_tags,
    remove_svg_tag,
    replace_gradient_tags,
)


@click.command()
@click.argument(
    "function", type=click.Choice(["encode", "decode", "both"]), required=True
)
@click.argument("input_text", required=False)
def main(function, input_text):
    if not input_text or input_text == "-":
        click.echo("Enter prompt:", nl=False)
        input_text = sys.stdin.read().strip()
        if not input_text:
            click.echo("No prompt provided.", err=True)
            sys.exit(1)

    svg_simplifier = SVGSimplifier()
    output_text_2 = None
    if function == "encode":
        output_text = svg_simplifier.encode(input_text)
    elif function == "decode":
        output_text = svg_simplifier.decode(input_text)
    elif function == "both":
        output_text = svg_simplifier.encode(input_text)
        output_text_2 = svg_simplifier.decode(output_text)
    else:
        raise ValueError(f"Unknown function: {function}")
    print(output_text)
    if output_text_2:
        print(output_text_2)


class SVGSimplifier:
    def __init__(self):
        pass

    def encode(self, svg_str):
        result_svg_1 = remove_defs_tags(svg_str)
        result_svg_2 = replace_gradient_tags(result_svg_1)
        result_svg_3 = apply_g_attributes_to_children(result_svg_2)
        result_svg = remove_svg_tag(result_svg_3)
        return result_svg

    def decode(self, sim_svg_str, logger=None, ignore_errors=True):
        sim_svg_str = lstrip_until_path(sim_svg_str)
        svg_str = add_svg_header(
            sim_svg_str, apply_view_box=True, ignore_errors=ignore_errors, logger=logger
        )
        return svg_str


def rstrip_from_last_newline(s: str) -> str:
    idx = s.rfind("\n")
    return s[:idx] if idx != -1 else s  # cut off from last "\n"


def lstrip_until_path(s: str) -> str:
    idx = s.find("<path")
    return s[idx:] if idx != -1 else s  # return unchanged if "<path" not found


def add_svg_header(
    simplifed_svg, apply_view_box=False, ignore_errors=False, logger=None
):
    paths = None
    attrs = None

    # NOTE: if the generation does not end, we add the endings or remove error commands
    err_msgs = []
    success = False
    temp_svg = None
    valid_simplifed_svg = ""
    for upper_idx in range(len(simplifed_svg), -1, -1):
        # NOTE: try without the closing tag `"/>\n`
        try:
            temp_svg = simplifed_svg[:upper_idx]
            temp_svg = "<svg>\n" + temp_svg + "\n</svg>"
            paths, attrs = svgstr2paths(temp_svg)
            success = True
            valid_simplifed_svg = simplifed_svg[:upper_idx]
            break
        except Exception as e:
            err_msg = f"Error parsing SVG: {e}\nSVG: {temp_svg}"
            if not ignore_errors:
                raise RuntimeError(err_msg)
            err_msgs.append(err_msg)

        # NOTE: try to remove the closing tag `"/>\n`
        try:
            temp_svg = simplifed_svg[:upper_idx] + '"/>\n'
            temp_svg = "<svg>\n" + temp_svg + "\n</svg>"
            paths, attrs = svgstr2paths(temp_svg)
            success = True
            valid_simplifed_svg = simplifed_svg[:upper_idx] + '"/>\n'
            break
        except Exception as e:
            err_msg = f"Error parsing SVG: {e}\nSVG: {temp_svg}"
            if not ignore_errors:
                raise RuntimeError(err_msg)
            err_msgs.append(err_msg)
    if not success and logger is not None:
        logger.error("\n".join(err_msgs))

    svg_str = (
        f'<svg xmlns="http://www.w3.org/2000/svg">\n' + valid_simplifed_svg + "</svg>"
    )
    if paths is not None and attrs is not None and apply_view_box:
        try:
            view_box = get_new_bounding_box(paths, attrs)
            view_box_str = " ".join([str(x) for x in view_box])
            svg_str = (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box_str}">\n'
                + valid_simplifed_svg
                + "</svg>"
            )
        except Exception as e:
            if not ignore_errors:
                raise RuntimeError(f"Error getting new bounding box for SVG: {e}")

    return svg_str


if __name__ == "__main__":
    main()
