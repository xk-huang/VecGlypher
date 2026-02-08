"""
pip install cairosvg
"""

import json
from io import BytesIO
from pathlib import Path

import cairosvg
from iopath.common.file_io import PathManager
from iopath.fb.storage import StoragePathHandler

pathmgr = PathManager()
pathmgr.register_handler(StoragePathHandler())


def main():
    metadata_chunk_path = "storage://workspace/svg_word_generation/pangrams/metadata/00000.jsonl"
    metadata_chunk_path_ = pathmgr.get_local_path(metadata_chunk_path)
    print(f"remote path: {metadata_chunk_path}")
    print(f"logal path: {metadata_chunk_path_}")

    sample = None
    with open(metadata_chunk_path_, "r") as f:
        for line in f:
            sample = json.loads(line)
            break
    text_svg = sample["text_svg"]
    text_svg_buf = BytesIO(text_svg.encode("utf-8"))

    output_dir = Path("misc")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = str(output_dir / "svg2img.jpg")
    cairosvg.svg2png(bytestring=text_svg_buf.getvalue(), write_to=output_path)
    print(f"SVG saved to {output_path}")


if __name__ == "__main__":
    main()
