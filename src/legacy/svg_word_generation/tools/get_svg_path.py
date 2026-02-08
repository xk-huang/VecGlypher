"""
python src/svg_word_generation/tests/render_text_with_fonttools.py

blackrenderer 'data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf' 'Hello!' misc/bkrd.svg --font-size 1000

python src/svg_word_generation/tools/compare_font_svg_path.py misc/font_tools.svg
python src/svg_word_generation/tools/compare_font_svg_path.py misc/bkrd.svg -x 3000
"""

import click
from svgpathtools import Path, svg2paths2
from svgpathtools.parser import parse_transform
from svgpathtools.path import transform, translate


@click.command()
@click.argument("input_svg_path", type=click.Path(exists=True))
@click.option("--translate_x", "-x", default=0, type=float)
@click.option("--translate_y", "-y", default=0, type=float)
def main(input_svg_path, translate_x, translate_y):
    paths, attributes, svg_attributes = svg2paths2(input_svg_path)
    path = paths[0]
    attr = attributes[0]
    for path, attr in zip(paths, attributes):
        transform_str = attr.get("transform")
        # transform_str = transform_str.replace('-1', '1')
        # transform_str = transform_str.replace('1900', '0')
        matrix = parse_transform(transform_str)
        new_path = transform(path, matrix)
        matrix = parse_transform(f"translate({translate_x}, {translate_y})")
        new_path = transform(new_path, matrix)
        # print(path.d())
        # print(matrix)
        print(new_path.d())


if __name__ == "__main__":
    main()
