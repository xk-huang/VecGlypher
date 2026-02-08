import click
import svgwrite
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont


def text_to_svg(
    font_path="data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf",
    text,
    fill="#ff5722",
    filename="misc/font_tools.svg",
    font_size=1000,
):  # units-per-em you want to export with
    font = TTFont(font_path)
    original_scale = font["head"].unitsPerEm
    print(f"scale: {original_scale} -> {font_size}")
    scale = font_size / original_scale

    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()

    asc = font["hhea"].ascent  # ← 1) define it here
    desc = font["hhea"].descent  # (optional) for full height

    x_cursor = 0
    glyphs = []
    for ch in text:
        gname = cmap.get(ord(ch))
        if gname is None:
            raise ValueError(f"No glyph for {repr(ch)} in font")
        pen = SVGPathPen(glyph_set)
        glyph_set[gname].draw(pen)
        glyphs.append((pen.getCommands(), x_cursor))
        x_cursor += glyph_set[gname].width * scale
    # overall svg
    dwg = svgwrite.Drawing(
        filename,
        viewBox=f"0 0 {x_cursor} {asc-desc}",
    )
    # add each glyph
    for d, x in glyphs:
        dwg.add(
            dwg.path(
                d=d,
                fill=fill,
                transform=f"translate({x} {asc * scale}) scale({scale} -{scale})",  # ← 2) flip first, then shift
            )
        )
    dwg.save()
    print("Wrote", filename)


@click.command()
@click.argument("font_path", type=click.Path(exists=True))
@click.argument("text", type=str)
@click.option("--font-size", default=1000, type=float)
def main(font_path, text, font_size):
    text_to_svg(font_path, text, font_size=font_size)
    print("Done!")


if __name__ == "__main__":
    main()
