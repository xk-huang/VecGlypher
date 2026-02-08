"""
python -m google_font_processor.write_metadata_tags_to_jsonl
"""

import json
from concurrent.futures import as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import click

import tqdm

from .read_metadata_tags import load_font_metadata, load_group_tags, load_tags


def split_font_variants(metadata):
    """
    Split the metadata into separate entries for each font variant.
    """
    metadata_list = []
    name = metadata["name"]

    num_font_variants = len(metadata["fonts"])
    metadata["num_font_variants"] = num_font_variants

    for font_data in metadata["fonts"]:
        font_data_name = font_data["name"]

        if font_data_name != name:
            raise ValueError(
                f"Font data name {font_data_name} does not match metadata name {name}"
            )

        metadata_ = metadata.copy()
        metadata_.pop("fonts")
        for k, v in font_data.items():
            if k != "name":
                if k in metadata:
                    raise ValueError(f"Duplicate key {k} in font data and metadata")
                metadata_[k] = v
        metadata_list.append(metadata_)

    return metadata_list


def json_dumps_list(metadata_list):
    return [json.dumps(metadata) for metadata in metadata_list]


def load_metadata_for_font(font_dir_name, font_dir, group_tags_df):
    """
    **Standalone worker** — must accept *only* picklable args so the
    ThreadPoolExecutor can hand it to a worker thread.
    """
    font_metadata_path = Path(font_dir) / font_dir_name / "METADATA.pb"
    if not font_metadata_path.exists():
        print(f"Font metadata file does not exist: {font_metadata_path}")
        # signal failure
        return None, font_dir_name

    try:
        metadata = load_font_metadata(font_metadata_path, font_dir_name)
        metadata = load_group_tags(metadata, group_tags_df)
        split_metadata = split_font_variants(metadata)
        split_metadata = json_dumps_list(split_metadata)
        return split_metadata, None  # success
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load metadata for {font_dir_name}: {e}")
        # collect the failing name so we can log it later
        return None, font_dir_name


@click.command()
@click.option("--num_workers", type=int, default=None)
def main(num_workers: int | None = None):
    """Spawn a thread-pool and gather the results."""
    # ------------------------------------------------------------------
    # 0.  Prep
    # ------------------------------------------------------------------
    group_tags_root = "data/google_fonts/tags/all/"
    version = "new"
    print(f"Loading tags from {group_tags_root} (version {version}) ...")
    group_tags_df = load_tags(group_tags_root, version)

    font_dir = "data/google_fonts/"
    font_dir_path = Path(font_dir)
    font_dir_name_list = [p.name for p in font_dir_path.iterdir()]
    print(f"font dir: {font_dir_path}; number of fonts: {len(font_dir_name_list)}")
    print(
        f"font family dir name list: {font_dir_name_list[:5]} ... {font_dir_name_list[-5:]}"
    )

    output_dir = Path("data/google_font_processor/")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file_path = output_dir / "google_font_metadata.jsonl"
    print(f"output file path: {output_file_path}")

    font_dir_name_ = font_dir_name_list[0]
    metadata_ = load_metadata_for_font(font_dir_name_, font_dir, group_tags_df)
    print(f"Loaded metadata for {font_dir_name_}: {metadata_}")

    # ------------------------------------------------------------------
    # 1.  Parallel section
    # ------------------------------------------------------------------
    n_workers = num_workers or max(cpu_count() * 2, 8)  # good default for I/O
    metadata_jsonl_list: list[str] = []
    failed_font_list: list[str] = []

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                load_metadata_for_font,
                font_dir_name,
                font_dir,
                group_tags_df,
            ): font_dir_name
            for font_dir_name in font_dir_name_list
        }

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            metadata_jsonl_list_, failed_name = future.result()
            if metadata_jsonl_list_ is not None:
                metadata_jsonl_list.extend(metadata_jsonl_list_)
            elif failed_name is not None:
                failed_font_list.append(failed_name)

    # ------------------------------------------------------------------
    # 2.  Write once at the end
    # ------------------------------------------------------------------
    with open(output_file_path, "w") as f:
        for metadata_jsonl in metadata_jsonl_list:
            f.write(metadata_jsonl + "\n")

    # ------------------------------------------------------------------
    # 3.  Optional: show failures
    # ------------------------------------------------------------------
    total_font = len(metadata_jsonl_list) + len(failed_font_list)
    print(f"Number of typefaces (font family): {len(font_dir_name_list)}")
    print(f"Number of fonts: {total_font}")
    print(f"Number of fonts that were processed: {len(metadata_jsonl_list)}")
    if failed_font_list:
        print(f"Number of fonts that could not be processed: {len(failed_font_list)}")
        print("Failed fonts:")
        print(failed_font_list)


if __name__ == "__main__":
    main()
