import math
import re
import sys

import click

# Match a single path command OR a number (incl. scientific notation)
CMD_RE = r"[AaCcHhLlMmQqSsTtVvZz]"
NUM_RE = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?"
TOKEN_RE = re.compile(f"({CMD_RE}|{NUM_RE})")

NUM_ONLY_RE = re.compile(f"^{NUM_RE}$")


def round_half_up(x, ndigits=0):
    factor = 10**ndigits
    return math.floor(x * factor + 0.5) / factor


def round_num_str(s: str, decimals) -> str:
    """Round a numeric string to nearest int and return as string."""
    if decimals is None:
        decimals = 0
    rounded_num = round_half_up(float(s), decimals)

    if decimals == 0:
        rounded_num = int(rounded_num)

    return str(rounded_num)


def minify_path_d(d: str, decimals: int | None = None) -> str:
    """
    Tokenize path data into commands and numbers.
    - Round numbers
    - Remove all spaces
    - Insert commas between consecutive numbers (safe for all commands, incl. A/a).
    """
    tokens = TOKEN_RE.findall(d)
    out = []
    for i, tok in enumerate(tokens):
        if NUM_ONLY_RE.match(tok):
            tok = round_num_str(tok, decimals)
        out.append(tok)
        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if NUM_ONLY_RE.match(out[-1]) and NUM_ONLY_RE.match(nxt):
                out.append(",")
    return "".join(out)


def minify_transform(
    transform: str, decimals: int | None = None, skip_space=False
) -> str:
    """
    Compact transforms: translate/scale/rotate/skewX/skewY/matrix.
    - Round numbers
    - Remove all spaces
    - Use commas between args in each (...) block
    """

    def repl(m):
        inner = m.group(1)
        nums = re.findall(NUM_RE, inner)
        nums = [round_num_str(n, decimals) for n in nums]
        return "(" + ",".join(nums) + ")"

    t = re.sub(r"\(([^()]*)\)", repl, transform.strip())
    if skip_space:
        return t.replace(" ", "")  # drop any remaining spaces between functions
    return t


def round_rm_space_svg_path(svg_text: str, decimals: int | None = None) -> str:
    """
    Process an entire SVG string:
    - Minify every d="..." it finds (so multiple <path> tags are handled).
    - Minify every transform="...".
    Preserves original quoting (single/double).
    """

    # d="..."/d='...'
    def d_repl(m):
        quote, d = m.group(1), m.group(2)
        return f"d={quote}{minify_path_d(d, decimals)}{quote}"

    svg_text = re.sub(r'd=(["\'])(.*?)\1', d_repl, svg_text, flags=re.DOTALL)

    # transform="..."/transform='...'
    def tr_repl(m):
        quote, t = m.group(1), m.group(2)
        return f"transform={quote}{minify_transform(t, decimals)}{quote}"

    svg_text = re.sub(r'transform=(["\'])(.*?)\1', tr_repl, svg_text, flags=re.DOTALL)

    # Optional: normalize whitespace between attributes
    svg_text = re.sub(r">\s+<", "><", svg_text)  # collapse inter-tag whitespace
    return svg_text


@click.command()
@click.option("--input_svg_file", "-i", default=None, type=str)
def main(input_svg_file):
    if input_svg_file is None:
        # load from stdin
        print("Loading from stdin...")
        svg = sys.stdin.read()
    with open(input_svg_file, "r") as f:
        svg = f.read()
    print(f"==================\n{svg}\n==================")
    print(round_rm_space_svg_path(svg))


if __name__ == "__main__":
    main()
