"""
streamlit run src/tools/sft_data_visualizer.py --server.port 8443

The format of jsonl:
- "instruction" / "prompt"
- "label" (output format of llamafactory) / "output" (from instruction data of llamafactory format)
- "predict"

"""

import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from svg_glyph_gen_v2.svg_simplifier import SVGSimplifier


svg_simplifier = SVGSimplifier()


def _load_jsonl_data(file_path) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data_ = json.loads(line.strip())

                    # add metadata to sort data by different keys
                    metadata = data_.pop("metadata", None)
                    metadata = json.loads(metadata) if metadata else {}
                    data_.update(metadata)

                    identifier = data_.get("identifier", None)
                    content_str = data_.get("content_str", None)
                    if identifier and content_str:
                        data_.update(
                            {"identifier_content_str": f"{identifier}_{content_str}"}
                        )

                    data.append(data_)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return []


def load_jsonl_data(file_path: str) -> List[Dict]:
    path = Path(file_path)
    # NOTE: special hack for storage_cli mount, sometimes we need to
    # call path.* multiple times to the results
    data = []
    for retry in range(3):
        try:
            if path.is_dir():
                # If the path is a directory, look for a JSONL file in it
                jsonl_files = list(path.glob("*.jsonl"))
                data = []
                for jsonl_file in jsonl_files:
                    data.extend(_load_jsonl_data(jsonl_file))
            else:
                # If the path is a file, load the JSONL data from it
                data = _load_jsonl_data(path)
            if data:
                break
        except Exception as e:
            st.warning(f"Error loading data: {str(e)}, retrying {retry+1} ...")
            time.sleep(1)
    return data


def load_jsonl_data_group(file_path_list_str):
    file_path_list = file_path_list_str.split("\n")
    file_path_list = [p for p in file_path_list if len(p) > 0]
    data = None
    predict_suffix_idx = 0
    short_file_path = remove_common_affixes(file_path_list)
    for idx, file_path in enumerate(file_path_list):
        if not data:
            data = load_jsonl_data(file_path)
            for data_item in data:
                data_item[f"predict_path"] = short_file_path[idx]
        else:
            # fmt: off
            new_data = load_jsonl_data(file_path)
            identifier_content_str2new_data_idx = {new_data[i]["identifier_content_str"]: i for i in range(len(new_data))}
            for data_item in data:
                identifier_content_str = data_item.get("identifier_content_str", None)
                if identifier_content_str is None:
                    continue
                if identifier_content_str in identifier_content_str2new_data_idx:
                    new_data_idx = identifier_content_str2new_data_idx[identifier_content_str]
                    new_data_item = new_data[new_data_idx]
                    data_item[f"predict{predict_suffix_idx}"] = new_data_item["predict"]
                    data_item[f"predict_path{predict_suffix_idx}"] = short_file_path[idx]
            # fmt: on
        predict_suffix_idx += 1
    return data


def remove_common_affixes(strings):
    if not strings:
        return strings

    # --- Common prefix ---
    prefix = os.path.commonprefix(strings)
    trimmed = [s[len(prefix) :] for s in strings]

    # --- Common suffix ---
    # reverse all, find common prefix, reverse back
    reversed_strings = [s[::-1] for s in trimmed]
    rev_prefix = os.path.commonprefix(reversed_strings)
    suffix = rev_prefix[::-1]

    result = [s[: -len(suffix)] if suffix else s for s in trimmed]
    return result


def decode_svg_if_needed(svg_text: str) -> str:
    """Extract SVG content, removing any thinking tags."""
    if svg_text.startswith("<svg") or svg_text.startswith("<?xml"):
        return svg_text
    try:
        decoded_svg_text = svg_simplifier.decode(svg_text)
    except Exception as e:
        return "<svg>" + svg_text + "</svg>"

    return decoded_svg_text.strip()


def main():
    st.set_page_config(
        page_title="SVG Font Visualization", page_icon="🔤", layout="wide"
    )

    st.text("🔤 SVG Font Visualization Tool")
    st.markdown("Compare predicted vs actual SVG font renderings")

    # File path input
    default_path = (
        "data/processed/sft/250813-oxford_5000-100_fonts/train-sample_100/00000.jsonl"
    )
    file_path = st.text_area("JSONL File Path (Split by newline):", value=default_path)
    other_metadata_path = st.text_input("Other Metadata Directory:", "")

    if not file_path:
        st.warning("Please provide a file path.")
        return

    # Initialize session state for caching data
    if "cached_data" not in st.session_state:
        st.session_state.cached_data = {}

    # Check if we need to reload data
    cache_key = f"{file_path}|{other_metadata_path}"
    needs_reload = (
        cache_key not in st.session_state.cached_data
        or st.session_state.cached_data[cache_key]["file_path"] != file_path
        or st.session_state.cached_data[cache_key]["other_metadata_path"]
        != other_metadata_path
    )

    if needs_reload:
        # Load data
        with st.spinner("Loading data..."):
            tic = time.time()
            data = load_jsonl_data_group(file_path)
            if other_metadata_path:
                other_metadata = load_jsonl_data(other_metadata_path)
            else:
                other_metadata = None
            delta_time = time.time() - tic

            # Cache the loaded data
            st.session_state.cached_data[cache_key] = {
                "file_path": file_path,
                "other_metadata_path": other_metadata_path,
                "data": data,
                "other_metadata": other_metadata,
                "load_time": delta_time,
                "entry_count": len(data),
            }
            st.sidebar.success(
                f"Loaded {len(data)} entries in {delta_time:.2f} seconds"
            )
    else:
        # Use cached data
        cached_info = st.session_state.cached_data[cache_key]
        data = cached_info["data"]
        other_metadata = cached_info["other_metadata"]
        st.sidebar.info(
            f"Using cached data: {cached_info['entry_count']} entries (loaded in {cached_info['load_time']:.2f}s)"
        )

    if not data:
        st.error("No data loaded. Please check the file path.")
        return

    use_other_metadata = False
    if other_metadata is not None:
        # check entries
        use_other_metadata = True
        for data_, metadata_ in zip(data, other_metadata):
            # NOTE: not sure when this key will be changed
            if data_["svg_path"] != metadata_["svg_path"]:
                use_other_metadata = False
                break
        if not use_other_metadata:
            st.error("Metadata does not match. Please check the metadata file path.")
            return
        for data_, other_metadata_ in zip(data, other_metadata):
            for key in other_metadata_:
                if key not in data_:
                    data_[key] = other_metadata_[key]

    # Sidebar controls
    with st.sidebar.expander("🔍 Filter Options", expanded=False):
        # Filter text input
        filter_text = st.text_input("Filter by text (case-insensitive):", "")

    # Match key and content input
    with st.sidebar.expander("🔍 Matchting Options", expanded=False):
        match_key = st.text_input("Match keys", "")
        match_content = st.text_input("Match content", "")

    char_counts = None
    # Character distribution
    if char_counts is None:
        entry = data[0]
        input_text_list = None
        for key in ["prompt", "instruction"]:
            if key in entry:
                input_text_list = [entry[key] for entry in data]
                break
        char_counts = {}
        if input_text_list:
            for char in input_text_list:
                char_counts[char] = char_counts.get(char, 0) + 1
    else:
        pass
    # Statistics section
    # if char_counts:
    #     st.text("📊 Statistics")
    #     str_list = []
    #     for char, count in sorted(char_counts.items()):
    #         char = char.replace("\n", " ")
    #         str_list.append(f"'{char}': {count}")
    #     str_list = ", ".join(str_list)
    #     st.text_area("Character distribution:", str_list, height=100)

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
                search_fields.append(prompt_text)
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

    if match_key and match_content:
        original_count = len(data)
        filtered_data = []

        for entry in data:
            # Check if filter text appears in various fields
            if match_key in entry:
                value = entry[match_key]
                if isinstance(value, bool):
                    value = str(value)
                if match_content.lower() != value.lower():
                    continue
                filtered_data.append(entry)
        data = filtered_data
        st.sidebar.success(f"Matched: {len(data)} of {original_count} entries")

    if not data:
        st.warning("No entries match the filter criteria.")
        return

    st.sidebar.header("🔧 Display Options")
    page_size = st.sidebar.slider("Entries per page", 5, 50, 10, 5)
    columns = st.sidebar.slider("Columns", 1, 30, 10)
    svg_height = st.sidebar.number_input("SVG Height", value=50)

    show_ground_truth = st.sidebar.checkbox("Show Ground Truth", True)
    show_prediction = st.sidebar.checkbox("Show Prediction", True)
    show_code = st.sidebar.checkbox("Show SVG", False)
    show_metadata = st.sidebar.checkbox("Show metadata", False)
    wrap_content = st.sidebar.checkbox("Warp content", True)

    # build sort keys
    sample = data[0]
    key_candidates = list(sample.keys())
    if "metadata" in key_candidates and isinstance(sample["metadata"], dict):
        key_candidates = list(key_candidates) + list(sample["metadata"].keys())
    st.sidebar.text_area("Available keys:", "\n".join(key_candidates), height=300)
    caption_key = st.sidebar.text_input("Caption key", key_candidates[0])

    # sort option
    sort_by_prompt = st.sidebar.checkbox("Sort by prompt", False)
    sort_keys = st.sidebar.text_input(
        "Sort keys (split by ,)", "font_family_dir_name,filename,content_str"
    )
    # check sort_keys
    sort_keys = sort_keys.split(",")
    sort_keys = [key.strip() for key in sort_keys if key.strip()]
    if set(sort_keys) - set(key_candidates):
        st.sidebar.error(f"Invalid sort keys: {set(sort_keys) - set(key_candidates)}")
        sort_keys = [key_candidates[0]]

    if sort_by_prompt:
        data = sorted(data, key=lambda x: tuple(x[k] for k in sort_keys))

    # Pagination
    num_pages = (len(data) + page_size - 1) // page_size

    # Initialize session state for current page if not exists
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    # Navigation controls
    st.header("📄 Navigation")

    # Direction arrows for page navigation
    # widget is stateless so far.
    col1, col2, col3 = st.columns([1, 2, 1])

    use_left_right_arrow = False
    with col1:
        if (
            st.button("◀", help="Previous page", use_container_width=True)
            and st.session_state.current_page > 1
        ):
            st.session_state.current_page -= 1
            use_left_right_arrow = True
    with col3:
        if (
            st.button("▶", help="Next page", use_container_width=True)
            and st.session_state.current_page < num_pages
        ):
            st.session_state.current_page += 1
            use_left_right_arrow = True
    with col2:
        if num_pages <= 1:
            st.text("No pages")
            new_page = 1
        else:
            new_page = st.slider(
                "Page (will not be updated when using left/right arrow)",
                min_value=1,
                max_value=num_pages,
                value=None,
            )
        if not use_left_right_arrow:
            st.session_state.current_page = new_page
        st.text(f"Page {st.session_state.current_page} of {num_pages}")

    # Page number input (sync with session state)
    if st.session_state.current_page > num_pages:
        st.session_state.current_page = num_pages

    start_idx = (st.session_state.current_page - 1) * page_size
    end_idx = min(start_idx + page_size, len(data))

    entries_info = f"Showing **{start_idx + 1}–{end_idx}** of **{len(data)}** entries\npage: { st.session_state.current_page} of {num_pages}"

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
                    label_svg = entry.get("output", "")
                    if not label_svg:
                        label_svg = entry.get("label", "")
                    if not label_svg:
                        label_svg = entry.get("svg", "")

                    if label_svg:
                        label_svg = decode_svg_if_needed(label_svg)
                        try:
                            tree = ET.ElementTree(ET.fromstring(label_svg))
                            root = tree.getroot()
                            view_box = root.attrib.get("viewBox", "0 0 100 100")
                            x, y, width, height = list(map(float, view_box.split(" ")))
                            svg_width = int(svg_height * width / height)
                        except Exception:
                            svg_width = svg_height
                        try:
                            st.image(
                                label_svg, use_container_width=False, width=svg_width
                            )
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")

                    input_text = entry.get(caption_key, "")
                    if not input_text:
                        continue
                    caption_style = (
                        "text-align: center; font-size: 12px; margin-top: 5px;"
                    )
                    caption_html = (
                        f'<div style="{caption_style}">'
                        f"<strong>'{input_text}'</strong></div>"
                    )
                    st.markdown(caption_html, unsafe_allow_html=True)

        # Predicted Row
        if show_prediction:
            st.text("🤖 Predicted SVGs")
            pred_cols = st.columns(len(row_entries))

            for col, entry in zip(pred_cols, row_entries):
                predicted_svg_list = []
                predicted_file_list = []
                if "predict" in entry:
                    predicted_svg_list.append(entry["predict"])
                    predicted_file_list.append(entry["predict_path"])
                predict_idx = 1
                while True:
                    if f"predict{predict_idx}" in entry:
                        predicted_svg_list.append(entry[f"predict{predict_idx}"])
                        predicted_file_list.append(entry[f"predict_path{predict_idx}"])
                        predict_idx += 1
                    else:
                        break
                with col:

                    for predicted_file, predicted_svg in zip(
                        predicted_file_list, predicted_svg_list
                    ):
                        predicted_svg = decode_svg_if_needed(predicted_svg)
                        try:
                            tree = ET.ElementTree(ET.fromstring(predicted_svg))
                            root = tree.getroot()
                            view_box = root.attrib.get("viewBox", "0 0 100 100")
                            x, y, width, height = list(map(float, view_box.split(" ")))
                            svg_width = int(svg_height * width / height)
                        except Exception:
                            svg_width = svg_height
                        try:
                            st.text(f"{predicted_file}")
                            st.image(
                                predicted_svg,
                                use_container_width=False,
                                width=svg_width,
                            )
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")

                    input_text = entry.get(caption_key, "")
                    if not input_text:
                        continue

                    caption_style = (
                        "text-align: center; font-size: 12px; margin-top: 5px;"
                    )
                    caption_html = (
                        f'<div style="{caption_style}">'
                        f"<strong>'{input_text}'</strong></div>"
                    )
                    st.markdown(caption_html, unsafe_allow_html=True)

        # Show SVG code if requested
        if show_code:
            row_num = row_start // columns + 1
            with st.expander(f"Show SVG Code for Row {row_num}"):
                for i, entry in enumerate(row_entries):
                    if not caption_key:
                        st.write("No caption key provided, skipping")
                        continue

                    input_text = entry[caption_key]
                    st.write(f"**Character '{input_text}':**")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Ground Truth:")
                        label_svg = entry.get("label", "")
                        if not label_svg:
                            label_svg = entry.get("output", "")
                        if label_svg:
                            st.code(label_svg, language="xml", wrap_lines=wrap_content)
                        else:
                            st.write("No ground truth available")

                    with col2:
                        st.write("Predicted:")
                        predicted_svg = entry.get("predict", "")
                        if predicted_svg:
                            st.code(
                                predicted_svg, language="xml", wrap_lines=wrap_content
                            )
                        else:
                            st.write("No prediction available")

                    if i < len(row_entries) - 1:
                        st.markdown("---")
        if show_metadata:
            row_num = row_start // columns + 1
            with st.expander(f"Show metadata for Row {row_num}"):
                for i, entry in enumerate(row_entries):
                    if not caption_key:
                        st.write("No caption key provided, skipping")
                        continue

                    input_text = entry.get(caption_key, "")
                    if not input_text:
                        continue
                    st.write(f"**Character '{input_text}':**")

                    st.code(str(entry), language="xml", wrap_lines=wrap_content)

                    if i < len(row_entries) - 1:
                        st.markdown("---")
        st.markdown("---")


if __name__ == "__main__":
    main()
