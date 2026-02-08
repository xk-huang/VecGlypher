"""
python src/svg_glyph_gen_v2_envato/count_alter_newline.py <input_path>
"""

import click, json, os, sys, tempfile

UNUSUAL = ("\u2028", "\u2029", "\u0085")  # LS, PS, NEL


@click.command()
@click.argument(
    "input_path", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    help="Write to this path instead of in-place.",
)
@click.option("--no-validate", is_flag=True, help="Skip JSONL validation.")
@click.option(
    "--backup/--no-backup",
    default=True,
    show_default=True,
    help="Keep .bak when doing in-place.",
)
def main(input_path, output, no_validate, backup):
    """Normalize unusual line terminators (LS/PS/NEL) in a JSONL file to LF."""
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        raw = f.read()

    counts = {
        "LS(\\u2028)": raw.count("\u2028"),
        "PS(\\u2029)": raw.count("\u2029"),
        "NEL(\\u0085)": raw.count("\u0085"),
    }
    click.echo(f"Found: {counts}")

    for row in raw.split("\n"):
        if any(u in row for u in UNUSUAL):
            click.echo(f"Unusual line: {row!r}\n", err=True)


if __name__ == "__main__":
    main()
