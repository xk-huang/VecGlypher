from pathlib import Path

import click
import svgwrite
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont


def text_to_svg(
    font_path_or_obj="data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf",
    text="test",
    output_path=None,
    font_size=1000,
    final_ascender=None,  # optional, if None, use font's ascender
    final_descender=None,  # optional, if None, use font's descender
):  # units-per-em you want to export with
    if not isinstance(font_path_or_obj, TTFont):
        font = TTFont(font_path_or_obj)
    else:
        font = font_path_or_obj

    upm = font["head"].unitsPerEm
    scale = font_size / upm

    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()

    asc = font["hhea"].ascent * scale  # ← 1) define it here
    desc = font["hhea"].descent * scale  # (optional) for full height

    if final_ascender is not None:
        asc = final_ascender
    if final_descender is not None:
        desc = final_descender

    # Pick a single space advance we can fall back to
    if "space" in glyph_set:
        space_advance = glyph_set["space"].width * scale
    else:  # 1/4 em as a last resort
        _space_advance = upm * 0.25
        space_advance = _space_advance * scale
        # print(
        #     f"no space glyph in font, using {_space_advance} -> {space_advance} "
        #     f"as space advance for font: {font_path_or_obj}"
        # )

    x_cursor = 0
    glyphs = []
    for ch in text:
        gname = cmap.get(ord(ch))
        if gname is None:
            if ch.isspace():  # space, tab, NBSP, etc.
                x_cursor += space_advance
                continue  # nothing to draw

            # option: use .notdef for other unmapped chars instead of skipping
            if ".notdef" in glyph_set:
                gname = ".notdef"
            else:
                # ignore completely
                continue
            raise ValueError(f"No glyph for {repr(ch)} in font {font_path_or_obj}")

        pen = SVGPathPen(glyph_set)
        glyph_set[gname].draw(pen)
        glyphs.append((pen.getCommands(), x_cursor))
        x_cursor += glyph_set[gname].width * scale
    # overall svg
    dwg = svgwrite.Drawing(
        filename=output_path,
        viewBox=f"0 0 {x_cursor} {asc-desc}",
    )
    # add each glyph
    for d, x in glyphs:
        if not d:
            continue
        dwg.add(
            dwg.path(
                d=d,
                transform=f"translate({x} {asc}) scale({scale} -{scale})",  # ← 2) flip first, then shift
            )
        )

    if output_path is not None:
        dwg.save()
        return

    svg_str = dwg.tostring()
    return svg_str


@click.command()
@click.argument(
    "font_path",
    type=click.Path(exists=True),
    default="data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf",
)
@click.argument("text", type=str, default="test")
@click.argument("output_path", type=str, default="misc/test.svg")
@click.option("--font-size", default=1000, type=float)
@click.option("--final-ascender", default=None, type=float)
@click.option("--final-descender", default=None, type=float)
def main(font_path, text, output_path, font_size, final_ascender, final_descender):
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    text_to_svg(
        font_path,
        text,
        output_path,
        font_size=font_size,
        final_ascender=final_ascender,
        final_descender=final_descender,
    )
    print(f"save svg to: {output_path}")


if __name__ == "__main__":
    main()
