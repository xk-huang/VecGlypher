from blackrenderer.render import renderText


def main():
    font_path = "misc/Honk[MORF,SHLN].ttf"
    output_path = "misc/ABC.svg"
    text = "ABC"
    renderText(font_path, text, output_path)  # or "output.svg"
    print(f"rendered text: {text} to {output_path}")


if __name__ == "__main__":
    main()
