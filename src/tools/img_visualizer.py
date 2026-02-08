"""
streamlit run src/tools/sft_data_visualizer.py --server.port 8445
"""

import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import streamlit as st

from PIL import Image

DEFAULT_DIR = "outputs/alphanumeric-ood_fonts-alphanumeric-train_f-train_c-qwen3_32b-full_sft-lr_5.0e-5-checkpoint-4376/raw_outputs/infer_decoded_render"
DEFAULT_PATTERN = "**/*"

st.set_page_config(page_title="Image Grid Viewer", layout="wide")


# ---------- Helpers ----------
def natural_key(s: str):
    # sort like humans: file2.png < file10.png
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


@st.cache_data(show_spinner=False)
def list_images(
    root: str, pattern: str, include_exts: List[str], sort_by: str, ascending: bool
) -> List[Path]:
    root_path = Path(root).expanduser().resolve()
    # Normalize pattern: if absolute, use as-is; else join with root
    search_root = root_path
    matches = list(search_root.glob(pattern))
    # Filter extensions if user typed a broad pattern
    if include_exts:
        exts = {e.lower() for e in include_exts}
        matches = [p for p in matches if p.is_file() and p.suffix.lower() in exts]
    # Sorting
    if sort_by == "Name (natural)":
        matches.sort(key=lambda p: natural_key(str(p)), reverse=not ascending)
    elif sort_by == "Modified time":
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=not ascending)
    else:
        matches.sort(key=lambda p: str(p).lower(), reverse=not ascending)
    return matches


def relative_caption(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p)


def chunk(iterable, n):
    # yield successive n-sized chunks
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


# ---------- Sidebar controls ----------
st.sidebar.title("Controls")

root_dir = st.sidebar.text_input("Root directory", value=DEFAULT_DIR)
pattern = st.sidebar.text_input(
    "Glob pattern (recursive supported)",
    value=DEFAULT_PATTERN,
    help="Examples: **/*.png, **/*.[jp][pn]g",
)
filter_text = st.sidebar.text_input("Filename filter (substring, optional)", value="")
show_caption = st.sidebar.checkbox("Use regex for filter", value=False)

sort_by = st.sidebar.selectbox("Sort by", ["Name (natural)", "Modified time"])
ascending = st.sidebar.checkbox("Ascending", value=True)

num_rows = st.sidebar.slider(
    "Rows per page", min_value=1, max_value=12, value=4, step=1
)
num_cols = st.sidebar.slider(
    "Columns per page", min_value=1, max_value=12, value=6, step=1
)
per_page = num_rows * num_cols

thumb_fit = st.sidebar.selectbox("Image fit", ["Fixed height", "Column width"])
thumb_height = None
if thumb_fit == "Fixed height":
    thumb_height = st.sidebar.number_input(
        "Thumbnail height (px)", min_value=32, max_value=1024, value=64, step=16
    )

show_full_paths = st.sidebar.checkbox("Show full path instead of relative", value=False)
exts = st.sidebar.multiselect(
    "Extensions to include",
    [".png", ".jpg", ".jpeg", ".webp", ".svg"],
    default=[".png"],
)

refresh = st.sidebar.button("Refresh list / clear cache")
if refresh:
    st.cache_data.clear()

show_caption = st.sidebar.checkbox("Show caption", value=True)
show_code = st.sidebar.checkbox("Show SVG Code", False)

# ---------- Load files ----------
if not root_dir.strip():
    st.error("Please provide a root directory.")
    st.stop()

root_path = Path(root_dir).expanduser()

if not root_path.exists():
    st.error(f"Directory not found: {root_path}")
    st.stop()

all_images = list_images(root_dir, pattern, exts, sort_by, ascending)

# Filter by substring/regex
if filter_text.strip():
    if show_caption:
        try:
            rx = re.compile(filter_text, re.IGNORECASE)
            all_images = [p for p in all_images if rx.search(str(p))]
        except re.error as e:
            st.warning(f"Regex error: {e}. Showing unfiltered results.")
    else:
        ft = filter_text.lower()
        all_images = [p for p in all_images if ft in str(p).lower()]

count = len(all_images)
st.title("🖼️ Image Grid Viewer")
st.write(
    f"**Root:** `{root_path}`  •  **Pattern:** `{pattern}`  •  **Found:** **{count}** file(s)"
)

if count == 0:
    st.info("No images matched. Try a different directory, pattern, or filter.")
    st.stop()

# ---------- Pagination ----------
num_pages = max(1, math.ceil(count / per_page))
if "page" not in st.session_state:
    st.session_state.page = 1

col_prev, col_page, col_next = st.columns([1, 2, 1])
use_left_right_arrow = False
with col_prev:
    if st.button("⬅️ Prev", use_container_width=True) and st.session_state.page > 1:
        st.session_state.page -= 1
        use_left_right_arrow = True
with col_next:
    if (
        st.button("Next ➡️", use_container_width=True)
        and st.session_state.page < num_pages
    ):
        st.session_state.page += 1
        use_left_right_arrow = True
with col_page:
    new_page = st.slider(
        "Page (will not be updated when using left/right arrow)",
        min_value=1,
        max_value=num_pages,
        step=1,
    )
    if not use_left_right_arrow:
        st.session_state.page = new_page

    start = (st.session_state.page - 1) * per_page
    end = min(start + per_page, count)
    page_paths = all_images[start:end]

    st.write(
        f"Showing **{start + 1}–{end}** of **{count}** (page {st.session_state.page} of {num_pages})"
    )

# ---------- Grid render ----------

for row_start in range(0, len(page_paths), num_cols):
    row_paths = page_paths[row_start : row_start + num_cols]

    cols = st.columns(len(row_paths))
    for col, p in zip(cols, row_paths):
        cap = str(p) if show_full_paths else relative_caption(p, root_path)
        with col:
            try:
                if thumb_height is not None:
                    if p.suffix.lower() == ".svg":
                        tree = ET.parse(p)
                        root = tree.getroot()
                        x, y, width, height = list(
                            map(float, root.attrib.get("viewBox").split(" "))
                        )

                    else:
                        img = Image.open(p)
                        width, height = img.size
                    thumb_width = int(thumb_height * width / height)
                    st.image(
                        str(p),
                        use_container_width=False,
                        width=thumb_width,
                    )
                else:
                    st.image(str(p), caption=cap, use_container_width=True)
            except Exception as e:
                print(f"Error loading {p}: {e}")

            if show_caption:
                caption = str(p) if show_full_paths else relative_caption(p, root_path)
                caption_style = "text-align: center; font-size: 12px; margin-top: 5px;"
                caption_html = (
                    f'<div style="{caption_style}">'
                    f"<strong>'{caption}'</strong></div>"
                )
                st.markdown(caption_html, unsafe_allow_html=True)

    if show_code:
        num_rows_code = row_start // num_cols + 1
        with st.expander(f"Show SVG Code for Row {num_rows_code}"):
            for i, entry in enumerate(row_paths):
                try:
                    with open(entry, "r") as f:
                        svg_code = f.read()
                except Exception as e:
                    st.error(f"Error reading SVG code: {e}")
                    continue

                st.code(svg_code, language="xml", wrap_lines=True)
