"""
streamlit run src/svg_word_generation/tools/svg_gen_visualizer.py --server.port 8443

The format of jsonl:
- "instruction" / "prompt"
- "label" (output format of llamafactory) / "output" (from instruction data of llamafactory format)
- "predict"

"""

import json
import re
from pathlib import Path
from typing import Dict, List

import streamlit as st


def _load_jsonl_data(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return []


def load_jsonl_data(file_path: str) -> List[Dict]:
    path = Path(file_path)
    if path.is_dir():
        print("Loading data from directory:", path)
        # If the path is a directory, look for a JSONL file in it
        jsonl_files = list(path.glob("*.jsonl"))
        data = []
        for jsonl_file in jsonl_files:
            data.extend(_load_jsonl_data(jsonl_file))
    else:
        # If the path is a file, load the JSONL data from it
        data = _load_jsonl_data(path)
    return data


def extract_character_from_prompt(prompt: str, remove_prefix_str: str = "") -> str:
    """Extract the character from the prompt text."""
    # Look for "Text content: X" pattern
    # match = re.search(r"Text content:\s*([^\n\r]+)", prompt)
    # if match:
    #     return match.group(1).strip()
    # return "Unknown"
    prompt = prompt.lstrip(remove_prefix_str)
    return prompt


def extract_svg_content(svg_text: str) -> str:
    """Extract SVG content, removing any thinking tags."""
    # Remove <think> tags and their content
    svg_text = re.sub(r"<think>.*?</think>", "", svg_text, flags=re.DOTALL)

    # Find the SVG content
    svg_match = re.search(r"(<\?xml.*?</svg>)", svg_text, flags=re.DOTALL)
    if svg_match:
        return svg_match.group(1).strip()

    # If no XML declaration, look for just the SVG tag
    svg_match = re.search(r"(<svg.*?</svg>)", svg_text, flags=re.DOTALL)
    if svg_match:
        return svg_match.group(1).strip()

    return svg_text.strip()


def render_svg(svg_content: str, width: int = 200, height: int = 200) -> str:
    """Render SVG content as HTML with specified dimensions."""
    if not svg_content:
        no_svg_div = (
            "<div style='border: 1px solid #ccc; width: 200px; "
            "height: 200px; display: flex; align-items: center; "
            "justify-content: center;'>No SVG</div>"
        )
        return no_svg_div

    # Clean the SVG content
    is_svg = ""
    clean_svg = extract_svg_content(svg_content)
    if not clean_svg.endswith("</svg>"):
        if not clean_svg.endswith("</svg>"):
            clean_svg = clean_svg + "</svg>"
        elif not clean_svg.endswith('"/>\n</svg>'):
            clean_svg = clean_svg + '"/>\n</svg>'
        is_svg = "fail end </svg>"

    # Wrap in a container div with specified dimensions
    container_style = (
        f"border: 1px solid #ddd; padding: 10px; width: {width}px; "
        f"height: {height}px; display: flex; align-items: center; "
        f"justify-content: center; background: white;"
    )
    return f'<div style="{container_style}">{is_svg}{clean_svg}</div>'


def main():
    st.set_page_config(
        page_title="SVG Font Visualization", page_icon="🔤", layout="wide"
    )

    st.text("🔤 SVG Font Visualization Tool")
    st.markdown("Compare predicted vs actual SVG font renderings")

    # File path input
    default_path = "misc/qwen-8b-ood_test_word.jsonl"
    file_path = st.text_input("JSONL File Path:", value=default_path)

    if not file_path:
        st.warning("Please provide a file path.")
        return

    # Load data
    with st.spinner("Loading data..."):
        data = load_jsonl_data(file_path)

    if not data:
        st.error("No data loaded. Please check the file path.")
        return

    # st.success(f"Loaded {len(data)} entries")

    # Sidebar controls
    st.sidebar.header("🔍 Filter Options")

    # Filter text input
    filter_text = st.sidebar.text_input("Filter by text (case-insensitive):", "")
    remove_prefix_str = st.sidebar.text_area(
        "Remove prefix (case-sensitive):",
        "system\nDesign SVG code for the given text content, follow the given the font design requirements. Do not use <text> element in generated SVG, use <path> instead.\nuser\n",
    )
    char_counts = None
    # Character distribution
    if char_counts is None:
        try:
            characters = [
                extract_character_from_prompt(entry["prompt"], remove_prefix_str)
                for entry in data
            ]
        except KeyError:
            characters = [
                extract_character_from_prompt(entry["instruction"], remove_prefix_str)
                for entry in data
            ]
        char_counts = {}
        for char in characters:
            char_counts[char] = char_counts.get(char, 0) + 1
    else:
        pass
    # Statistics section
    if char_counts:
        st.text("📊 Statistics")
        str_list = []
        for char, count in sorted(char_counts.items()):
            char = char.replace("\n", " ")
            str_list.append(f"'{char}': {count}")
        str_list = ", ".join(str_list)
        st.text_area("Character distribution:", str_list, height=100)

    # Additional info
    st.sidebar.markdown("---")
    st.sidebar.info(f"Total entries: {len(data)}")

    # Apply filter if text is provided
    if filter_text.strip():
        original_count = len(data)
        filtered_data = []

        for entry in data:
            # Check if filter text appears in various fields
            search_fields = []

            # Add prompt/instruction text
            if "prompt" in entry:
                search_fields.append(entry["prompt"])
            if "instruction" in entry:
                search_fields.append(entry["instruction"])

            # Add character extracted from prompt
            try:
                prompt_text = entry.get("prompt", entry.get("instruction", ""))
                character = extract_character_from_prompt(
                    prompt_text, remove_prefix_str
                )
                search_fields.append(character)
            except Exception:
                pass

            # Add SVG content fields
            if "label" in entry:
                search_fields.append(entry["label"])
            if "output" in entry:
                search_fields.append(entry["output"])
            if "predict" in entry:
                search_fields.append(entry["predict"])

            # Check if filter text appears in any field (case-insensitive)
            matches = any(
                filter_text.lower() in str(field).lower()
                for field in search_fields
                if field
            )
            if matches:
                filtered_data.append(entry)

        data = filtered_data
        st.sidebar.success(f"Filtered: {len(data)} of {original_count} entries")

    if not data:
        st.warning("No entries match the filter criteria.")
        return

    st.sidebar.header("🔧 Display Options")
    page_size = st.sidebar.slider("Entries per page", 5, 50, 10, 5)
    columns = st.sidebar.slider("Columns", 1, 10, 5)
    svg_width = st.sidebar.number_input("SVG Width", value=200)
    svg_height = st.sidebar.number_input("SVG Height", value=200)
    show_code = st.sidebar.checkbox("Show SVG Code", False)
    show_captions = st.sidebar.checkbox("Show prompt", True)
    show_prediction = st.sidebar.checkbox("Show Prediction", True)
    show_ground_truth = st.sidebar.checkbox("Show Ground Truth", True)
    sort_by_prompt = st.sidebar.checkbox("Sort by prompt", False)

    if sort_by_prompt:
        try:
            data = sorted(data, key=lambda x: x["prompt"])
        except KeyError:
            data = sorted(data, key=lambda x: x["instruction"])

    # Pagination
    num_pages = (len(data) + page_size - 1) // page_size

    # Initialize session state for current page if not exists
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    # Navigation controls
    st.sidebar.header("📄 Navigation")

    # Direction arrows for page navigation
    col1, col2, col3 = st.sidebar.columns([1, 2, 1])

    with col1:
        prev_disabled = st.session_state.current_page <= 1
        if st.button("◀", help="Previous page", disabled=prev_disabled):
            st.session_state.current_page = max(1, st.session_state.current_page - 1)
            st.rerun()

    with col3:
        next_disabled = st.session_state.current_page >= num_pages
        if st.button("▶", help="Next page", disabled=next_disabled):
            st.session_state.current_page = min(
                num_pages, st.session_state.current_page + 1
            )
            st.rerun()

    # Page number input (sync with session state)
    if st.session_state.current_page > num_pages:
        st.session_state.current_page = num_pages
    current_page = st.sidebar.number_input(
        "Page", 1, num_pages, st.session_state.current_page, 1, key="page_input"
    )

    # Update session state if page input changes
    if current_page != st.session_state.current_page:
        st.session_state.current_page = current_page

    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, len(data))

    entries_info = f"Showing **{start_idx + 1}–{end_idx}** of **{len(data)}** entries\npage: {current_page} of {num_pages}"
    st.sidebar.markdown(entries_info)

    # Get entries for current page
    page_entries = data[start_idx:end_idx]

    # Display entries in rows
    for row_start in range(0, len(page_entries), columns):
        row_entries = page_entries[row_start : row_start + columns]

        # Ground Truth Row
        if show_ground_truth:
            st.text("✅ Ground Truth SVGs")
            gt_cols = st.columns(len(row_entries))

            for col, entry in zip(gt_cols, row_entries):
                with col:
                    try:
                        character = extract_character_from_prompt(
                            entry["prompt"], remove_prefix_str
                        )
                    except KeyError:
                        character = extract_character_from_prompt(
                            entry["instruction"], remove_prefix_str
                        )

                    label_svg = entry.get("label", "")
                    if not label_svg:
                        label_svg = entry.get("output", "")

                    if label_svg:
                        st.markdown(
                            render_svg(label_svg, svg_width, svg_height),
                            unsafe_allow_html=True,
                        )
                    else:
                        no_gt_style = (
                            f"border: 1px solid #ccc; width: {svg_width}px; "
                            f"height: {svg_height}px; display: flex; "
                            f"align-items: center; justify-content: center; "
                            f"background: #f0f0f0;"
                        )
                        st.markdown(
                            f'<div style="{no_gt_style}">No Ground Truth</div>',
                            unsafe_allow_html=True,
                        )

                    if show_captions:
                        caption_style = (
                            "text-align: center; font-size: 12px; margin-top: 5px;"
                        )
                        caption_html = (
                            f'<div style="{caption_style}">'
                            f"<strong>'{character}'</strong></div>"
                        )
                        st.markdown(caption_html, unsafe_allow_html=True)

        # Predicted Row
        if show_prediction:
            st.text("🤖 Predicted SVGs")
            pred_cols = st.columns(len(row_entries))

            for col, entry in zip(pred_cols, row_entries):
                with col:
                    try:
                        character = extract_character_from_prompt(
                            entry["prompt"], remove_prefix_str
                        )
                    except KeyError:
                        character = extract_character_from_prompt(
                            entry["instruction"], remove_prefix_str
                        )

                    predicted_svg = entry.get("predict", "")

                    if predicted_svg:
                        st.markdown(
                            render_svg(predicted_svg, svg_width, svg_height),
                            unsafe_allow_html=True,
                        )
                    else:
                        no_pred_style = (
                            f"border: 1px solid #ccc; width: {svg_width}px; "
                            f"height: {svg_height}px; display: flex; "
                            f"align-items: center; justify-content: center; "
                            f"background: #f0f0f0;"
                        )
                        st.markdown(
                            f'<div style="{no_pred_style}">No Prediction</div>',
                            unsafe_allow_html=True,
                        )

                    if show_captions:
                        caption_style = (
                            "text-align: center; font-size: 12px; margin-top: 5px;"
                        )
                        caption_html = (
                            f'<div style="{caption_style}">'
                            f"<strong>'{character}'</strong></div>"
                        )
                        st.markdown(caption_html, unsafe_allow_html=True)

        # Show SVG code if requested
        if show_code:
            row_num = row_start // columns + 1
            with st.expander(f"Show SVG Code for Row {row_num}"):
                for i, entry in enumerate(row_entries):
                    try:
                        character = extract_character_from_prompt(
                            entry["prompt"], remove_prefix_str
                        )
                    except KeyError:
                        character = extract_character_from_prompt(
                            entry["instruction"], remove_prefix_str
                        )

                    st.write(f"**Character '{character}':**")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Ground Truth:")
                        label_svg = entry.get("label", "")
                        if not label_svg:
                            label_svg = entry.get("output", "")
                        if label_svg:
                            clean_label = extract_svg_content(label_svg)
                            st.code(clean_label, language="xml")
                        else:
                            st.write("No ground truth available")

                    with col2:
                        st.write("Predicted:")
                        predicted_svg = entry.get("predict", "")
                        if predicted_svg:
                            clean_pred = extract_svg_content(predicted_svg)
                            st.code(clean_pred, language="xml")
                        else:
                            st.write("No prediction available")

                    if i < len(row_entries) - 1:
                        st.markdown("---")

        st.markdown("---")


if __name__ == "__main__":
    main()
