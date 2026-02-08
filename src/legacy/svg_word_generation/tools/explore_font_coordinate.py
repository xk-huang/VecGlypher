from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont


def glyph_svg_centered(
    font: TTFont,
    char: str,
    *,
    center: str = None,
    margin=0,
    scale_to=None,
    round_to_int=False,
    offset_x=0,
    offset_y=0,
):
    """
    Return (svg_path_d, (xmin, ymin, xmax, ymax)) for *char* in *font_path*.

    *center*:
        "glyph" – centre the glyph on (0,0)
        "font"  – centre the font's em‑square on (0,0)

    *round_to_int*:
        If True, round all numeric values to integers
    """
    gset = font.getGlyphSet()
    gname = font.getBestCmap()[ord(char)]

    # --- original glyph bounds (font coords, Y‑up) ---------------------------
    bb_pen = BoundsPen(gset)
    gset[gname].draw(bb_pen)
    xmin, ymin, xmax, ymax = bb_pen.bounds

    # --- choose translation centre ------------------------------------------
    if center is None:
        cx, cy = 0, 0
    elif center == "glyph":
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
    elif center == "font":  # "font"
        xMin, yMin, xMax, yMax = (
            font["head"].xMin,
            font["head"].yMin,
            font["head"].xMax,
            font["head"].yMax,
        )
        cx, cy = (xMin + xMax) / 2, (yMin + yMax) / 2
    else:
        raise ValueError(f"Unknown center: {center}")

    # --- draw with translate + scale ----------------------------------------
    svg_pen = SVGPathPen(gset)

    # scale
    if scale_to is None:
        scale = 1
    else:
        scale = scale_to / font["head"].unitsPerEm

    # Combine translate and scale into single transformation matrix
    # First translate by (-cx, -cy), then scale by (scale, scale)
    # Combined matrix: [scale, 0, 0, scale, -cx*scale, -cy*scale]
    tpen = TransformPen(
        svg_pen, (scale, 0, 0, -scale, -cx * scale + offset_x, cy * scale - offset_y)
    )
    gset[gname].draw(tpen)
    d = svg_pen.getCommands()

    # --- bbox in transformed coords ------------------------------------------
    new_xmin = (xmin - cx) * scale + offset_x
    new_xmax = (xmax - cx) * scale + offset_x
    new_ymin = (ymin - cy) * scale - offset_y
    new_ymax = (ymax - cy) * scale - offset_y

    new_ymin, new_ymax = -new_ymax, -new_ymin  # SVG Y is down, not up

    new_bbox = (
        new_xmin - margin,
        new_ymin - margin,
        new_xmax + margin,
        new_ymax + margin,
    )

    # --- apply rounding if requested -----------------------------------------
    if round_to_int:
        import re

        # Round all numbers in the SVG path data
        def round_match(match):
            return str(round(float(match.group())))

        d = re.sub(r"-?\d+\.?\d*", round_match, d)

        # Round bounding box coordinates
        new_bbox = tuple(round(coord) for coord in new_bbox)

    return d, new_bbox


margin = 10
character = "I"
center = None
font = TTFont(
    "/mnt/workspace/data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf"
)
scale_to = 2000
for center in ["glyph"]:
    for offset_idx, character in enumerate("achq"):
        for round_to_int in [True]:
            # print(
            #     f"Center: {center}, Character: {character}, Round to int: {round_to_int}"
            # )
            d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
                font,
                character,
                center=center,
                margin=margin,
                scale_to=scale_to,
                round_to_int=round_to_int,
                offset_x=offset_idx * scale_to / 2,
            )
            print(d)
            print("")
            viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
            svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'
            # print(svg)
            # print("---")

    print("\n")


for center in ["font"]:
    for offset_idx, character in enumerate("achq"):
        for round_to_int in [True]:
            # print(
            #     f"Center: {center}, Character: {character}, Round to int: {round_to_int}"
            # )
            d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
                font,
                character,
                center=center,
                margin=margin,
                scale_to=scale_to,
                round_to_int=round_to_int,
                offset_x=offset_idx * scale_to / 2,
                offset_y=scale_to,
            )
            print(d)
            print("")
            viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
            svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'
            # print(svg)
            # print("---")

    print("\n")


for center in ["font"]:
    for offset_idx, character in enumerate("achq"):
        for round_to_int in [True]:
            # print(
            #     f"Center: {center}, Character: {character}, Round to int: {round_to_int}"
            # )
            d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
                font,
                character,
                center=center,
                margin=margin,
                scale_to=scale_to,
                round_to_int=round_to_int,
                # offset_x=offset_idx * scale_to / 2,
                # offset_y=scale_to,
            )
            print(d)
            print("")
            viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
            svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'
            print(svg)
            print("---")

    print("\n")

font_paths = [
    "/mnt/workspace/data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf",
    "/mnt/workspace/data/google_fonts/ofl/licorice/Licorice-Regular.ttf",
]
fonts = [TTFont(f) for f in font_paths]
center = "font"
rount_to_int = True
characters = "a"
for font in fonts:
    for character in characters:
        d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
            font,
            character,
            center=center,
            margin=margin,
            scale_to=scale_to,
            round_to_int=round_to_int,
            # offset_x=offset_idx * scale_to / 2,
            # offset_y=scale_to,
        )
        print(d)
        print("")
        viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
        svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'
        # print(svg)
        # print("---")


font_paths = [
    "/mnt/workspace/data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf",
    "/mnt/workspace/data/google_fonts/ofl/licorice/Licorice-Regular.ttf",
]
fonts = [TTFont(f) for f in font_paths]
center = "font"
rount_to_int = True
characters = "a"
for font in fonts:
    for character in characters:
        d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
            font,
            character,
            center=center,
            margin=margin,
            scale_to=scale_to,
            round_to_int=round_to_int,
            # offset_x=offset_idx * scale_to / 2,
            # offset_y=scale_to,
        )
        print(d)
        print("")
        viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
        svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'
        # print(svg)
        # print("---")

################## grid

font_paths = [
    "/mnt/workspace/data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf",
    "/mnt/workspace/data/google_fonts/ofl/licorice/Licorice-Regular.ttf",
    "/mnt/workspace/data/google_fonts/ofl/handlee/Handlee-Regular.ttf",
]
fonts = [TTFont(f) for f in font_paths]
center = None
rount_to_int = True
characters = "h"
scale_to = 2048
margin = 0
round_to_int = True

base_offset_x = 0
base_offset_y = 0

offset_idx = 0
for font in fonts:
    for character in characters:
        d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
            font,
            character,
            center=center,
            margin=margin,
            scale_to=scale_to,
            round_to_int=round_to_int,
            offset_x=offset_idx * scale_to + base_offset_x,
            offset_y=base_offset_y,
        )
        offset_idx += 1
        print(d)
        print("")
        viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
        svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'
        # print(svg)
        # print("---")

offset_idx = 0
for font in fonts:
    for character in characters:
        d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
            font,
            character,
            center=center,
            margin=margin,
            scale_to=scale_to,
            round_to_int=round_to_int,
            offset_x=base_offset_x,
            offset_y=offset_idx * scale_to + base_offset_y,
        )
        offset_idx += 1
        print(d)
        print("")
        viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
        svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'


font_paths = [
    "/mnt/workspace/data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf",
    "/mnt/workspace/data/google_fonts/ofl/licorice/Licorice-Regular.ttf",
    "/mnt/workspace/data/google_fonts/ofl/handlee/Handlee-Regular.ttf",
]
fonts = [TTFont(f) for f in font_paths]
center = "glyph"
rount_to_int = True
characters = "h"
scale_to = 2048
margin = 0
round_to_int = True

base_offset_x = scale_to / 2
base_offset_y = scale_to / 2

offset_idx = 0
for font in fonts:
    for character in characters:
        d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
            font,
            character,
            center=center,
            margin=margin,
            scale_to=scale_to,
            round_to_int=round_to_int,
            offset_x=offset_idx * scale_to + base_offset_x,
            offset_y=base_offset_y,
        )
        offset_idx += 1
        print(d)
        print("")
        viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
        svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'
        # print(svg)
        # print("---")

offset_idx = 0
for font in fonts:
    for character in characters:
        d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
            font,
            character,
            center=center,
            margin=margin,
            scale_to=scale_to,
            round_to_int=round_to_int,
            offset_x=base_offset_x,
            offset_y=offset_idx * scale_to + base_offset_y,
        )
        offset_idx += 1
        print(d)
        print("")
        viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
        svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'


font_paths = [
    "/mnt/workspace/data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf",
    "/mnt/workspace/data/google_fonts/ofl/licorice/Licorice-Regular.ttf",
    "/mnt/workspace/data/google_fonts/ofl/handlee/Handlee-Regular.ttf",
]
fonts = [TTFont(f) for f in font_paths]
center = "font"
rount_to_int = True
characters = "h"
scale_to = 2048
margin = 0
round_to_int = True

base_offset_x = scale_to / 2
base_offset_y = scale_to / 2

offset_idx = 0
for font in fonts:
    for character in characters:
        d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
            font,
            character,
            center=center,
            margin=margin,
            scale_to=scale_to,
            round_to_int=round_to_int,
            offset_x=offset_idx * scale_to + base_offset_x,
            offset_y=base_offset_y,
        )
        offset_idx += 1
        print(d)
        print("")
        viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
        svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'
        # print(svg)
        # print("---")

offset_idx = 0
for font in fonts:
    for character in characters:
        d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
            font,
            character,
            center=center,
            margin=margin,
            scale_to=scale_to,
            round_to_int=round_to_int,
            offset_x=base_offset_x,
            offset_y=offset_idx * scale_to + base_offset_y,
        )
        offset_idx += 1
        print(d)
        print("")
        viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
        svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'


"""
M0 0v-2048h2048v2048z
M2048 0v-2048h2048v2048z
M0 -2048v-2048h2048v2048z
M4096 0v-2048h2048v2048z
M0 -4096v-2048h2048v2048z
"""

## different char
font_paths = [
    # "/mnt/workspace/data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf",
    "/mnt/workspace/data/google_fonts/ofl/licorice/Licorice-Regular.ttf",
    # "/mnt/workspace/data/google_fonts/ofl/handlee/Handlee-Regular.ttf",
]
fonts = [TTFont(f) for f in font_paths]
center = None
rount_to_int = True
characters = "ahg"
scale_to = 2048
margin = 0
round_to_int = True

base_offset_x = 0
base_offset_y = 0

offset_idx = 0
for font in fonts:
    for character in characters:
        d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
            font,
            character,
            center=center,
            margin=margin,
            scale_to=scale_to,
            round_to_int=round_to_int,
            offset_x=offset_idx * scale_to + base_offset_x,
            offset_y=base_offset_y,
        )
        offset_idx += 1
        print(d)
        print("")
        viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
        svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'
        # print(svg)
        # print("---")

offset_idx = 0
for font in fonts:
    for character in characters:
        d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
            font,
            character,
            center=center,
            margin=margin,
            scale_to=scale_to,
            round_to_int=round_to_int,
            offset_x=base_offset_x,
            offset_y=offset_idx * scale_to + base_offset_y,
        )
        offset_idx += 1
        print(d)
        print("")
        viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
        svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'
print("=================")


font_paths = [
    # "/mnt/workspace/data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf",
    "/mnt/workspace/data/google_fonts/ofl/licorice/Licorice-Regular.ttf",
    # "/mnt/workspace/data/google_fonts/ofl/handlee/Handlee-Regular.ttf",
]
fonts = [TTFont(f) for f in font_paths]
center = "glyph"
rount_to_int = True
characters = "ahg"
scale_to = 2048
margin = 0
round_to_int = True

base_offset_x = scale_to / 2
base_offset_y = scale_to / 2

offset_idx = 0
for font in fonts:
    for character in characters:
        d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
            font,
            character,
            center=center,
            margin=margin,
            scale_to=scale_to,
            round_to_int=round_to_int,
            offset_x=offset_idx * scale_to + base_offset_x,
            offset_y=base_offset_y,
        )
        offset_idx += 1
        print(d)
        print("")
        viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
        svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'
        # print(svg)
        # print("---")

offset_idx = 0
for font in fonts:
    for character in characters:
        d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
            font,
            character,
            center=center,
            margin=margin,
            scale_to=scale_to,
            round_to_int=round_to_int,
            offset_x=base_offset_x,
            offset_y=offset_idx * scale_to + base_offset_y,
        )
        offset_idx += 1
        print(d)
        print("")
        viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
        svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'
print("=================")


font_paths = [
    # "/mnt/workspace/data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf",
    "/mnt/workspace/data/google_fonts/ofl/licorice/Licorice-Regular.ttf",
    # "/mnt/workspace/data/google_fonts/ofl/handlee/Handlee-Regular.ttf",
]
fonts = [TTFont(f) for f in font_paths]
center = "font"
rount_to_int = True
characters = "ahg"
scale_to = 2048
margin = 0
round_to_int = True

base_offset_x = scale_to / 2
base_offset_y = scale_to / 2

offset_idx = 0
for font in fonts:
    for character in characters:
        d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
            font,
            character,
            center=center,
            margin=margin,
            scale_to=scale_to,
            round_to_int=round_to_int,
            offset_x=offset_idx * scale_to + base_offset_x,
            offset_y=base_offset_y,
        )
        offset_idx += 1
        print(d)
        print("")
        viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
        svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'
        # print(svg)
        # print("---")

offset_idx = 0
for font in fonts:
    for character in characters:
        d, (xmin, ymin, xmax, ymax) = glyph_svg_centered(
            font,
            character,
            center=center,
            margin=margin,
            scale_to=scale_to,
            round_to_int=round_to_int,
            offset_x=base_offset_x,
            offset_y=offset_idx * scale_to + base_offset_y,
        )
        offset_idx += 1
        print(d)
        print("")
        viewbox = f"{xmin} {ymin} {xmax - xmin} {ymax - ymin}"
        svg = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n<path d="{d}" />\n</svg>'
print("=================")
