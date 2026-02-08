"""
Rely on SVGO
yarn global add svgo

font_path="data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf"
# font_path="data/google_fonts/ofl/rubikglitchpop/RubikGlitchPop-Regular.ttf"
python src/svg_glyph_gen_v2/render_text_with_fonttools.py $font_path test misc/test.svg

python src/svg_glyph_gen_v2/tests/apply_path_transform_and_round.py
"""

import pathlib
import re
import subprocess

from svgpathtools import svg2paths, svgstr2paths, wsvg
from svgpathtools.parser import parse_path, parse_transform
from svgpathtools.path import Path, transform, translate


TRANSFORM_SUFFIX = "transformed"
OPTIM_TRANSFORM_SUFFIX = "optim_transformed"
ROUND_OPTIM_TRANSFORM_SUFFIX = "round_optim_transformed"


# Control variables for applying transforms
def main():

    input_svg_path = pathlib.Path("./misc/test.svg")

    run(input_svg_path, apply_scale=True, apply_translate=False)
    run(input_svg_path, apply_scale=True, apply_translate=True)

    output_path_a = get_output_path(
        input_svg_path, TRANSFORM_SUFFIX, apply_scale=True, apply_translate=False
    )
    output_path_b = get_output_path(
        input_svg_path, TRANSFORM_SUFFIX, apply_scale=True, apply_translate=True
    )
    print(f"compare output paths: {output_path_a} and {output_path_b}")
    print(compare_svg_paths(output_path_a, output_path_b))

    output_path_a = get_output_path(
        input_svg_path, OPTIM_TRANSFORM_SUFFIX, apply_scale=True, apply_translate=False
    )
    output_path_b = get_output_path(
        input_svg_path, OPTIM_TRANSFORM_SUFFIX, apply_scale=True, apply_translate=True
    )
    print(f"compare output paths: {output_path_a} and {output_path_b}")
    print(compare_svg_paths(output_path_a, output_path_b))


def run(input_svg_path, apply_scale, apply_translate):

    input_svg_path = pathlib.Path(input_svg_path)

    if apply_translate is True and apply_scale is False:
        raise ValueError(
            "Applying translate without scale may result in unexpected behavior"
        )

    input_svg_text = input_svg_path.read_text()
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

        if not actual_tfm_parts:
            continue  # no transforms to apply

        actual_tfm_string = " ".join(actual_tfm_parts)

        # svgpathtools gives you a 3×3 matrix (a, b, c, d, e, f)
        # that follows the SVG spec's right-to-left multiplication order.
        tfm = parse_transform(actual_tfm_string)

        # -----  apply it -----
        p = transform(p, tfm)  # new Path object with every point moved
        paths[i] = p

        # -----  update attributes -----
        a["d"] = p.d(rel=True)  # new 'd'

        paths[i] = a["d"]

        # Set the transform attribute to unapplied transforms, or remove if none
        if unapplied_tfm_parts:
            a["transform"] = " ".join(unapplied_tfm_parts)
        else:
            a.pop("transform", None)

    output_svg_path = get_output_path(
        input_svg_path, TRANSFORM_SUFFIX, apply_translate, apply_scale
    )
    wsvg(
        paths,
        attributes=attrs,
        svg_attributes=svg_attr,
        filename=str(output_svg_path),
        viewbox=svg_attr["viewBox"],
    )

    print(f"========== Print path for {output_svg_path} ==========")
    for i in range(len(paths)):
        path_str = paths[i]
        path_str = "M" + path_str[1:]
        print(path_str)
    print(f"========== Print path for {output_svg_path} ==========")

    optim_output_svg_path = get_output_path(
        input_svg_path, OPTIM_TRANSFORM_SUFFIX, apply_translate, apply_scale
    )
    command = [
        "svgo",
        "-i",
        str(output_svg_path),
        "-o",
        str(optim_output_svg_path),
        "--config",
        "src/svgo_config/svgo.config.mjs",
    ]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise TimeoutError("Error running svgo:", result.stderr)
    else:
        print(f"svgo stdout: {result.stdout}")

    round_optim_output_svg_path = get_output_path(
        input_svg_path, ROUND_OPTIM_TRANSFORM_SUFFIX, apply_translate, apply_scale
    )
    command = [
        "svgo",
        "-i",
        str(output_svg_path),
        "-o",
        str(round_optim_output_svg_path),
        "--config",
        "src/svgo_config/svgo.config.round.mjs",
    ]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise TimeoutError("Error running svgo:", result.stderr)
    else:
        print(f"svgo stdout: {result.stdout}")

    print("compare input and output SVGs:")
    print(compare_svg_paths(input_svg_path, output_svg_path))

    print("compare output and optimized output SVGs:")
    print(compare_svg_paths(output_svg_path, optim_output_svg_path))


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


def get_output_path(input_svg_path, base_suffix, apply_translate, apply_scale):
    suffix = get_suffix(base_suffix, apply_translate, apply_scale)
    print(f"suffix: {suffix}")
    output_svg_path = input_svg_path.with_name(f"{input_svg_path.stem}.{suffix}.svg")
    return output_svg_path


def get_suffix(base_suffix, apply_translate, apply_scale):
    suffix = base_suffix
    if apply_translate:
        suffix += "_translate"
    if apply_scale:
        suffix += "_scale"
    return suffix


if __name__ == "__main__":
    main()
