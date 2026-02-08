"""
font_path=data/google_fonts/ofl/playwriteustradguides/PlaywriteUSTradGuides-Regular.ttf
text='fa f-a'
python src/svg_glyph_gen_v2/render_text_with_fonttools_v2.py ${font_path} ${text}
python src/svg_glyph_gen_v2/render_text_with_fonttools.py ${font_path} ${text}

Cursive/script fonts need OpenType shaping (GSUB/GPOS): ligatures, kerning, cursive
attachment (“curs”), contextual alternates (“calt”), Arabic init/medi/fina forms, mark
positioning, etc. Use HarfBuzz to turn text, a positioned glyph run (glyph IDs + per-glyph
x/y advances & offsets), then draw those glyphs at the shaped positions.
"""

from pathlib import Path

import click
import svgwrite

# pip install uharfbuzz
import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont


def _normalize_features(features):
    """
    Accepts:
      - dict like {"kern":1, "liga":1}
      - iterable of tags: ("kern","liga","calt")
      - iterable of "tag=value" strings: ("kern=1","liga=1")
      - iterable of (tag, value) tuples: (("ss01",1), ("liga",1))
    Returns a dict {tag:int(value)} suitable for hb.shape.
    """
    if not features:
        return {}
    if isinstance(features, dict):
        return {str(k): int(v) for k, v in features.items()}
    out = {}
    for f in features:
        if isinstance(f, str):
            if "=" in f:
                tag, val = f.split("=", 1)
                out[tag.strip()] = int(val)
            else:
                out[f] = 1
        else:
            # assume (tag, value)
            tag, val = f
            out[str(tag)] = int(val)
    return out


def text_to_svg_shaped(
    font_path="data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf",
    text="test",
    output_path="misc/test_shaped.svg",
    font_size=1000,
    # Toggle features as you like; curs/mark/mkmk matter for joining & diacritics
    features=("kern", "liga", "clig", "calt", "rlig", "curs", "mark", "mkmk"),
    # Optional variable font axes, e.g. {"wght": 700, "wdth": 100}
    variations=None,
    # Optional script/dir/lang overrides. If None, HarfBuzz will guess.
    direction=None,  # "LTR" | "RTL"
    language=None,  # e.g. "ar"
    script=None,  # e.g. "Arab"
):
    tt = TTFont(font_path)
    upm = tt["head"].unitsPerEm
    scale = font_size / upm

    glyph_set = tt.getGlyphSet()
    asc = tt["hhea"].ascent * scale
    desc = tt["hhea"].descent * scale

    # --- HarfBuzz shape ---
    fontdata = Path(font_path).read_bytes()
    hb_face = hb.Face(fontdata)
    hb_font = hb.Font(hb_face)
    hb_font.scale = (upm, upm)  # work in font units

    if variations:
        # Accept dict or list of (axis,val)
        if isinstance(variations, dict):
            hb_font.set_variations(variations)
        else:
            hb_font.set_variations(dict(variations))

    buf = hb.Buffer()
    buf.add_str(text)
    if direction or language or script:
        if direction:
            buf.direction = direction.lower()  # "rtl"/"ltr"
        if language:
            buf.language = language
        if script:
            buf.script = script
    else:
        buf.guess_segment_properties()

    feat_dict = _normalize_features(features)
    hb.shape(hb_font, buf, feat_dict)  # <-- dict, not list

    infos = buf.glyph_infos
    positions = buf.glyph_positions

    # HarfBuzz gives advances/offsets in font units (y+ upwards).
    # We keep a pen position in font units; convert to SVG units at the end.
    x_pen = 0
    y_pen = 0

    # Collect (path_d, x_px, y_px) for each glyph
    drawn = []
    for info, pos in zip(infos, positions):
        gid = info.codepoint
        gname = tt.getGlyphName(gid)

        pen = SVGPathPen(glyph_set)
        glyph_set[gname].draw(pen)
        d = pen.getCommands()
        if not d:
            # zero-width or empty glyph (e.g., control)
            x_pen += pos.x_advance
            y_pen += pos.y_advance
            continue

        # Glyph origin (font units), then offsets from GPOS
        gx = x_pen + pos.x_offset
        gy = y_pen + pos.y_offset

        # Convert to user units (SVG). We'll scale paths with scale, flip Y.
        x_px = gx * scale
        y_px = gy * scale

        drawn.append((d, x_px, y_px))

        x_pen += pos.x_advance
        y_pen += pos.y_advance

    total_width_px = (x_pen * scale) if x_pen else 0.0
    total_height_px = asc - desc

    # --- Draw SVG ---
    dwg = svgwrite.Drawing(
        filename=output_path,
        viewBox=f"0 0 {max(total_width_px, 1)} {max(total_height_px, 1)}",
    )

    for d, x_px, y_px in drawn:
        # IMPORTANT: transforms in SVG are applied right-to-left.
        # We want: (scale then flip Y), then translate to baseline.
        # asc*scale puts baseline at +asc; add shaped y offset (y_px).
        dwg.add(
            dwg.path(
                d=d,
                transform=f"translate({x_px} {(asc + y_px)}) scale({scale} {-scale})",
            )
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        dwg.save()
        return

    return dwg.tostring()


@click.command()
@click.argument(
    "font_path",
    type=click.Path(exists=True),
    default="data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf",
)
@click.argument("text", type=str, default="test")
@click.argument("output_path", type=str, default="misc/test_shaped.svg")
@click.option("--font-size", default=1000, type=float)
def main(font_path, text, output_path, font_size):
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    text_to_svg_shaped(font_path, text, output_path, font_size=font_size)
    print(f"save svg to: {output_path}")


if __name__ == "__main__":
    main()
