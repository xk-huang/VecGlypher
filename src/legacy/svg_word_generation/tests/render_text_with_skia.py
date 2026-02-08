from io import BytesIO

from ..custom_render import renderTextToObj


def main():

    font_path = "data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf"
    text = "The quick brown fox jumps over the lazy dog"
    backend = "skia"

    buf = BytesIO()
    renderTextToObj(font_path, text, buf, backendName=backend)
    buf.seek(0)
    text_svg = buf.read().decode("utf-8")
    breakpoint()


if __name__ == "__main__":
    main()
