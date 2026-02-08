# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-

import json
from pathlib import Path
from pprint import pformat, pprint

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

from fontTools.ttLib import TTFont
from genads.svg_vtg.google_font_processor.utils import local2storage
from iopath.common.file_io import PathManager
from iopath.fb.storage import StoragePathHandler

pathmgr = PathManager()
pathmgr.register_handler(StoragePathHandler())


def text_to_svg(
    text: str,
    font: TTFont,
    font_size: int = 72,  # px (EM-square in the final SVG)
    fill: str = "#000",
    x0: float = 0,
    y0: float = 0,
) -> str:
    """
    Convert *text* rendered with *ttf_path* into an <svg> string.

    Returns SVG markup that you can save or embed directly.
    """
    # --- load the font and basic metrics ---
    glyph_set = font.getGlyphSet()  # outline access
    cmap = font.getBestCmap()  # Unicode → glyph-name
    upm = font["head"].unitsPerEm  # units per EM
    scale = font_size / upm  # scale outlines to px

    x_cursor = x0  # running pen position
    svg_paths = []

    # --- build one <path> per glyph ---
    for ch in text:
        gname = cmap.get(ord(ch))
        if gname is None:  # glyph missing in font
            continue

        glyph = glyph_set[gname]

        # 1. capture the raw commands
        raw_pen = SVGPathPen(glyph_set)
        glyph.draw(raw_pen)

        # 2. replay them through a TransformPen to
        #    * scale to font_size
        #    * flip Y (font Y axis ↑, SVG Y axis ↓)
        #    * translate horizontally by current x advance
        trans_pen = SVGPathPen(glyph_set)
        transform = (
            scale,
            0,
            0,
            -scale,  # negative to flip Y
            x_cursor,
            y0,
        )  # translate
        tpen = TransformPen(trans_pen, transform)
        glyph.draw(tpen)

        svg_paths.append(f'<path d="{trans_pen.getCommands()}" fill="{fill}" />')

        # advance cursor by glyph width (optionally add kerning here)
        adv_width, _ = font["hmtx"][gname]
        x_cursor += adv_width * scale

    width = x_cursor
    height = font_size  # simple baseline-to-cap proxy

    # --- wrap all paths in a minimal SVG root ---
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 {-font_size} {width} {font_size}">\n'
        + "\n".join(svg_paths)
        + "\n</svg>"
    )
    return svg


def main() -> None:
    gfont_metadata_jsonl_path = (
        "workspace/google_font_processor/google_font_metadata.jsonl"
    )
    gfont_metadata_jsonl_path = Path(gfont_metadata_jsonl_path)
    gfont_metadata_jsonl_path = local2storage(gfont_metadata_jsonl_path)

    gfont_metadata_list = []
    with pathmgr.open(gfont_metadata_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            gfont_metadata_list.append(json.loads(line))

    gfont_name2metadata = {
        gfont_metadata["name"]: gfont_metadata for gfont_metadata in gfont_metadata_list
    }
    # take "Honk" as an example
    gfont_metadata = gfont_name2metadata["Honk"]

    # load font metadata
    font_base_dir = "workspace/google_fonts/ofl"
    font_dir_name = gfont_metadata["font_dir_name"]

    # get the font file name for each font
    font_file_data_list = gfont_metadata["fonts"]
    font_file_data = font_file_data_list[0]
    font_full_name = font_file_data["fullName"]
    font_file_name = font_file_data["filename"]

    font_file_path = Path(font_base_dir) / font_dir_name / font_file_name
    print(f"Loading font {font_full_name} from {font_file_path}")
    pprint(gfont_metadata)

    # load font
    if not pathmgr.exists(local2storage(font_file_path)):
        raise FileNotFoundError(f"Font file {font_file_path} not found.")

    with pathmgr.open(local2storage(font_file_path), "rb") as f:
        ttfont = TTFont(f)

    text = "llama"
    font_size = 200
    svg_code = text_to_svg(text, ttfont, font_size=font_size)

    output_dir = "misc"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_svg_path = output_dir / "svg_word-fonttools.svg"
    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write(svg_code)

    print(f"SVG saved to {output_svg_path}")


if __name__ == "__main__":
    main()
