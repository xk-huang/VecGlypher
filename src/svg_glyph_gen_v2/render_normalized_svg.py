"""
python -m src.svg_glyph_gen_v2.render_normalized_svg
"""

import os
import pathlib
import re
import subprocess
from io import StringIO
from types import SimpleNamespace
from xml.dom.minidom import parse as md_xml_parse

import click
from svgpathtools import disvg, svg2paths, svgstr2paths
from svgpathtools.parser import parse_transform
from svgpathtools.path import transform
from svgpathtools.paths2svg import big_bounding_box

from .render_text_with_fonttools import text_to_svg
from .round_rm_space_svg import round_rm_space_svg_path


def apply_transform_to_svg(
    input_svg_file, output_svg_file, *, apply_scale, apply_translate
):
    input_svg_file = pathlib.Path(input_svg_file)
    input_svg_text = input_svg_file.read_text()
    output_svg = apply_transform_to_svg_str(
        input_svg_text,
        apply_scale=apply_scale,
        apply_translate=apply_translate,
        output_svg_file=None,
    )
    if output_svg is None:
        raise RuntimeError("Failed to apply transform to SVG but failed")
    with open(output_svg_file, "w") as f:
        f.write(output_svg)


def apply_transform_to_svg_v1(
    input_svg_file, output_svg_file, *, apply_scale, apply_translate
):
    input_svg_file = pathlib.Path(input_svg_file)
    input_svg_text = input_svg_file.read_text()
    apply_transform_to_svg_str(
        input_svg_text,
        apply_scale=apply_scale,
        apply_translate=apply_translate,
        output_svg_file=output_svg_file,
        direct_save=True,
    )


def apply_transform_to_svg_str(
    input_svg_text,
    *,
    apply_scale,
    apply_translate,
    use_relative_path=True,
    output_svg_file=None,
    direct_save=False,
):
    if apply_translate is True and apply_scale is False:
        raise ValueError(
            "Applying translate without scale may result in unexpected behavior"
        )

    paths, attrs, svg_attr = svgstr2paths(input_svg_text, return_svg_attributes=True)

    for i, (p, a) in enumerate(zip(paths, attrs)):
        tfm_string = a.get("transform")
        if not tfm_string:
            continue  # nothing to bake-in for this path

        # Validate transform string format: translate(x y) scale(sx sy)
        expected_pattern = r"^translate\([-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+\)\s+scale\([-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+\)$"
        if not re.match(expected_pattern, tfm_string.strip()):
            print(
                f"Warning: Transform string '{tfm_string}' does not match expected format 'translate(x y) scale(sx sy)'"
            )
            continue  # skip this path if format doesn't match

        # Parse the original transform string to extract translate and scale components
        translate_match = re.search(
            r"translate\(([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\)", tfm_string
        )
        scale_match = re.search(
            r"scale\(([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\)", tfm_string
        )

        # Build the actual transform string based on control variables
        actual_tfm_parts = []
        unapplied_tfm_parts = []

        if apply_translate and translate_match:
            tx, ty = translate_match.groups()
            actual_tfm_parts.append(f"translate({tx} {ty})")
        elif translate_match:
            tx, ty = translate_match.groups()
            unapplied_tfm_parts.append(f"translate({tx} {ty})")

        if apply_scale and scale_match:
            sx, sy = scale_match.groups()
            actual_tfm_parts.append(f"scale({sx} {sy})")
        elif scale_match:
            sx, sy = scale_match.groups()
            unapplied_tfm_parts.append(f"scale({sx} {sy})")

        if actual_tfm_parts:
            actual_tfm_string = " ".join(actual_tfm_parts)

            # svgpathtools gives you a 3×3 matrix (a, b, c, d, e, f)
            # that follows the SVG spec's right-to-left multiplication order.
            tfm = parse_transform(actual_tfm_string)

            # -----  apply it -----
            p = transform(p, tfm)  # new Path object with every point moved
            paths[i] = p
        else:
            print(f"Warning: No transform applied to path {i}, {len(actual_tfm_parts)}")

        # -----  update attributes -----
        a["d"] = p.d(rel=use_relative_path)  # new 'd'

        # Set the transform attribute to unapplied transforms, or remove if none
        if unapplied_tfm_parts:
            a["transform"] = " ".join(unapplied_tfm_parts)
        else:
            a.pop("transform", None)

    # Re-compute the bounding box
    view_box = get_new_bounding_box(paths, attrs)

    # [NOTE](xk): if paths are List[Path], then the saved path are in the absolute coordinate system
    # If you use List[str] instead, then when we compute the bounding box, it raises error
    # thus we need to add `viewbox` argument. But if svg_attributes is not None, then it will overwrite
    # the viewBox attribute. The argument is really confusing. I think it should be removed.
    for i, a in enumerate(attrs):
        paths[i] = a["d"]
    svg_attr["viewBox"] = f"{view_box[0]} {view_box[1]} {view_box[2]} {view_box[3]}"
    dwg = disvg(
        paths,
        attributes=attrs,
        svg_attributes=svg_attr,
        filename=str(output_svg_file),
        viewbox=view_box,
        paths2Drawing=(not direct_save),
    )
    if dwg is None:
        return None

    return dump_dwg_to_string(dwg)


def dump_dwg_to_string(dwg, pretty=False, indent=2):
    f = StringIO()
    dwg.write(f, pretty=pretty, indent=indent)

    # NOTE: rewind, so that we can read the string again
    f.seek(0)
    # NOTE: svgpathtools/paths2svg.py: re-open the svg, make the xml pretty, and save it again
    xmlstring = md_xml_parse(f).toprettyxml()
    return xmlstring


def get_new_bounding_box(paths, attrs):
    new_paths = []
    for p, a in zip(paths, attrs):
        _transform_str = a.get("transform")
        if _transform_str is None:
            new_paths.append(p)
        else:
            new_paths.append(transform(p, parse_transform(_transform_str)))

    xmin, xmax, ymin, ymax = big_bounding_box(new_paths)
    dx = xmax - xmin
    dy = ymax - ymin
    return xmin, ymin, dx, dy


def round_optim_svg(input_svg_file, output_svg_file):
    with open(input_svg_file, "r") as f:
        svg = f.read()
    processed_svg = round_rm_space_svg_path(svg)
    with open(output_svg_file, "w") as f:
        f.write(processed_svg)


# def round_optim_svg(input_svg_file, output_svg_file, round_decimal=True):
# if round_decimal:
#     config = "src/svgo_config/svgo.config.round.mjs"
# else:
#     config = "src/svgo_config/svgo.config.mjs"
# command = [
#     "svgo",
#     "-i",
#     str(input_svg_file),
#     "-o",
#     str(output_svg_file),
#     "--config",
#     config,
# ]
# result = subprocess.run(command, capture_output=False, text=True)
# if result.returncode != 0:
#     raise TimeoutError("Error running svgo:", result.stderr)


# def round_optim_svg_dir(input_svg_dir, output_svg_dir, round_decimal=True, logger=None):
#     if round_decimal:
#         config = "src/svgo_config/svgo.config.round.mjs"
#     else:
#         config = "src/svgo_config/svgo.config.mjs"

#     # NOTE(xk): SVGO processes directory in parallel:
#     # ref: https://github.com/svg/svgo/blob/9e2029f9b101da81ef612ff7606669c71faf9085/lib/svgo/coa.js#L310
#     command = [
#         "svgo",
#         "-rf",
#         str(input_svg_dir),
#         "-o",
#         str(output_svg_dir),
#         "--config",
#         config,
#     ]
#     if logger is not None:
#         logger.info("Running command: " + " ".join(command))
#     else:
#         print("Running command:", " ".join(command))

#     # NOTE(xk): to avoid js heap out of memory error, set max_old_space_size=8192
#     # ref: https://github.com/svg/svgo/issues/954#issuecomment-759068787
#     # "NODE_OPTIONS=--max_old_space_size=8192",
#     env = os.environ.copy()
#     env["NODE_OPTIONS"] = "--max_old_space_size=8192"
#     result = subprocess.run(
#         command,
#         capture_output=False,
#         text=True,
#         env=env,
#     )

#     if result.returncode != 0:
#         if logger is not None:
#             logger.error(f"Error running svgo: {result.stderr}")
#         raise TimeoutError("Error running svgo:", result.stderr)


def compare_paths(paths_a, paths_b, attr_a=None, attr_b=None):
    if len(paths_a) != len(paths_b):
        return False

    for idx, (path_a, path_b) in enumerate(zip(paths_a, paths_b)):
        attr_a_ = attr_a[idx] if attr_a else None
        attr_b_ = attr_b[idx] if attr_b else None
        path_a = _transform_path_from_attr(path_a, attr_a_)
        path_b = _transform_path_from_attr(path_b, attr_b_)
        if path_a != path_b:
            print(f"Path {idx} does not match\n{path_a}\n{path_b}")
            return False

    return True


def _transform_path_from_attr(path, attr=None):
    if attr is None:
        return path
    tfm_string = attr.get("transform")
    if not tfm_string:
        return path

    # Parse the original transform string to extract translate and scale components
    tfm = parse_transform(tfm_string)
    return transform(path, tfm)


def compare_svg_paths(svg_file_path_a, svg_file_path_b):
    paths_a, attr_a = svg2paths(svg_file_path_a)
    paths_b, attr_b = svg2paths(svg_file_path_b)

    return compare_paths(paths_a, paths_b, attr_a, attr_b)


@click.command()
@click.argument(
    "input_font_file",
    type=click.Path(exists=True),
    default="data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf",
)
@click.argument("text", type=str, default="Hello!")
@click.argument(
    "output_svg_file",
    type=click.Path(exists=False),
    default="misc/render_normalized_svg.svg",
)
@click.option(
    "--temp_dir",
    type=click.Path(exists=False),
    default="misc/temp_render_normalized_svg",
)
@click.option("--apply_scale", type=bool, default=True)
@click.option("--apply_translate", type=bool, default=True)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    input_font_file = args.input_font_file
    output_svg_file = args.output_svg_file
    temp_dir = args.temp_dir
    apply_scale = args.apply_scale
    apply_translate = args.apply_translate

    temp_dir = pathlib.Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    render_svg_path = temp_dir / "render_svg.svg"
    apply_transform_to_svg_path_v1 = temp_dir / "apply_transform_to_svg.v1.svg"
    apply_transform_to_svg_path = temp_dir / "apply_transform_to_svg.svg"

    text_to_svg(input_font_file, args.text, render_svg_path)

    apply_transform_to_svg(
        render_svg_path,
        apply_transform_to_svg_path,
        apply_scale=apply_scale,
        apply_translate=apply_translate,
    )
    apply_transform_to_svg_v1(
        render_svg_path,
        apply_transform_to_svg_path_v1,
        apply_scale=apply_scale,
        apply_translate=apply_translate,
    )

    round_optim_svg(apply_transform_to_svg_path, output_svg_file)
    print(f"Output SVG file: {output_svg_file}")


if __name__ == "__main__":
    main()
