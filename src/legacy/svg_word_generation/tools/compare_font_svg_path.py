import subprocess

import click


@click.command()
@click.option(
    "--font_path",
    type=click.Path(exists=True),
    default="data/google_fonts/ofl/roboto/Roboto[wdth,wght].ttf",
)
@click.option("--text", type=str, default="Hello!")
@click.option("--font-size", default=1000, type=int)
@click.option("--translate_x", "-x", default=0, type=float)
@click.option("--backend", default="svg", type=str)
def main(font_path, text, font_size, translate_x, backend):
    command = [
        "python",
        "src/svg_word_generation/tests/render_text_with_fonttools.py",
        font_path,
        text,
        f"--font-size={font_size}",
    ]
    run_command(command, False)

    command = [
        "blackrenderer",
        font_path,
        text,
        "misc/bkrd.svg",
        f"--font-size={font_size}",
        f"--backend={backend}",
    ]
    run_command(command, False)

    command = [
        "python",
        "src/svg_word_generation/tools/get_svg_path.py",
        "misc/font_tools.svg",
    ]
    run_command(command)
    print("\n")
    command = [
        "python",
        "src/svg_word_generation/tools/get_svg_path.py",
        "misc/bkrd.svg",
        "-x",
        str(translate_x),
    ]
    run_command(command)


def run_command(command, print_output=True):
    # print("Running command: ", " ".join(command))
    results = subprocess.run(command, check=True, capture_output=print_output)
    if results.returncode != 0:
        raise RuntimeError(f"Failed to run command: {command}")
    if print_output:
        print(results.stdout.decode("utf-8"))


if __name__ == "__main__":
    main()
