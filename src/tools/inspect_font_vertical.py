"""
python src/tools/inspect_font_vertical.py
"""

import click
from fontTools.ttLib import TTFont


@click.command()
@click.argument("font_path", type=click.Path(exists=True))
def main(font_path):
    font = TTFont(font_path)
    hhea = font["hhea"]

    ascender = hhea.ascent
    descender = hhea.descent
    line_gap = hhea.lineGap
    print("Ascender:", ascender)
    print("Descender:", descender)
    print("LineGap:", line_gap)

    os2 = font["OS/2"]
    typo_ascender = os2.sTypoAscender
    typo_descender = os2.sTypoDescender
    typo_linegap = os2.sTypoLineGap
    print("Typo Ascender:", typo_ascender)
    print("Typo Descender:", typo_descender)
    print("Typo LineGap:", typo_linegap)

    upm = font["head"].unitsPerEm
    print(f"Units per EM: {upm}")


if __name__ == "__main__":
    main()
