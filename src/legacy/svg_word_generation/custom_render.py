import os

import skia

from blackrenderer.backends.skia import SkiaSVGSurface
from blackrenderer.backends.svg import SVGSurface, writeSVGElements
from blackrenderer.render import (
    BlackRendererFont,
    buildGlyphLine,
    calcGlyphLineBounds,
    hb,
    insetRect,
    intRect,
    scaleRect,
)


# support bytes writing
class CustomSVGSurface(SVGSurface):
    def saveImage(self, path):
        if isinstance(path, str):
            super().saveImage(path)
        else:
            writeSVGElements(self._svgElements, self._viewBox, path)


class CustomSkiaSVGSurface(SkiaSVGSurface):
    def saveImage(self, path):
        if isinstance(path, str):
            super().saveImage(path)
        else:
            stream = skia.DynamicMemoryWStream()
            picture = self._pictures[-1]
            canvas = skia.SVGCanvas.Make(picture.cullRect(), stream)
            canvas.drawPicture(picture)
            del canvas  # hand holding skia-python with GC: it needs to go before stream
            stream.flush()
            path.write(stream.detachAsData())


SurfaceMapping = {
    "skia": CustomSkiaSVGSurface,
    "svg": CustomSVGSurface,
}


def renderTextToObj(
    fontPath,
    textString,
    outputPath,
    *,
    fontSize=250,
    margin=20,
    features=None,
    variations=None,
    paletteIndex=0,
    backendName=None,
    lang=None,
    script=None,
):
    if not isinstance(fontPath, BlackRendererFont):
        font = BlackRendererFont(fontPath)
    else:
        font = fontPath
    glyphNames = font.glyphNames

    scaleFactor = fontSize / font.unitsPerEm

    buf = hb.Buffer()
    buf.add_str(textString)
    buf.guess_segment_properties()

    if script:
        buf.script = script
    if lang:
        buf.language = lang
    if variations:
        font.setLocation(variations)
    palette = font.getPalette(paletteIndex)

    hb.shape(font.hbFont, buf, features)

    infos = buf.glyph_infos
    positions = buf.glyph_positions
    glyphLine = buildGlyphLine(infos, positions, glyphNames)
    bounds = calcGlyphLineBounds(glyphLine, font)
    bounds = scaleRect(bounds, scaleFactor, scaleFactor)
    bounds = insetRect(bounds, -margin, -margin)
    bounds = intRect(bounds)

    if backendName is None:
        backendName = "svg"

    surface = SurfaceMapping[backendName]()
    with surface.canvas(bounds) as canvas:
        canvas.scale(scaleFactor)
        for glyph in glyphLine:
            with canvas.savedState():
                canvas.translate(glyph.xOffset, glyph.yOffset)
                font.drawGlyph(glyph.name, canvas, palette=palette)
            canvas.translate(glyph.xAdvance, glyph.yAdvance)

    surface.saveImage(outputPath)
