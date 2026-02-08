"""
Write SVG to jsonl
```
python -m src.svg_word_generation.svg_word_gen_bkrd_chunk \
    --input_words_path data/processed/words_curation/pangrams.txt \
    --output_dir data/processed/svg_word_generation/pangrams
python -m src.svg_word_generation.svg_word_gen_bkrd_chunk \
    --input_words_path data/processed/words_curation/words-en-2000.txt \
    --output_dir data/processed/svg_word_generation/words-en-2000 \
    --num_workers 150
```

train-font train-word
```
python -m src.svg_word_generation.svg_word_gen_bkrd_chunk \
    --input_words_path data/processed/words_curation/words-en-2000.txt \
    --font_split_tsv data/processed/split_train_test_index/words-en-2000/fonts/train.tsv \
    --word_split_tsv data/processed/split_train_test_index/words-en-2000/words/train.tsv \
    --output_dir data/processed/svg_word_generation/words-en-2000-train_font-train_word \
    --num_workers 150
```

Input text
```
python -m src.svg_word_generation.svg_word_gen_bkrd_chunk \
    --input_text "a" \
    --output_dir data/processed/svg_word_generation/a \
    --num_workers 30
```
"""

import hashlib
import json
import multiprocessing
import os
import shutil
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import List

import click
import pandas as pd

import tqdm
from blackrenderer.render import BlackRendererFont, renderText
from numpy import choose

from .custom_render import renderTextToObj


def compute_hash(text: str) -> str:
    """Return first 20 hex chars of SHA‑256 hash for *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def get_font_file_name_text_hash(
    font_file_name: str, text: str, return_input: bool = False
):
    """Stable hash for the (font‑file, text) pair used as output filename."""
    input_text = f"{font_file_name}:{text}"
    hash_text = compute_hash(input_text)
    if return_input:
        return hash_text, input_text
    return hash_text


def read_words(word_path: Path) -> List[str]:
    with word_path.open("r", encoding="utf-8") as f:
        return f.read().splitlines()


@lru_cache()
def _load_font(path):
    return BlackRendererFont(path)


def load_font(gfont_metadata, font_base_dir):
    font_file_name: str = gfont_metadata["filename"]
    font_dir_name: str = gfont_metadata["font_dir_name"]

    font_path = font_base_dir / font_dir_name / font_file_name
    return _load_font(font_path)


def render_word_font(
    word: str,
    gfont_metadata: dict,
    font_path=None,
    font_base_dir=None,
    output_dir=None,
    save_svg: bool = False,
    save_png: bool = False,
    backend: str = "skia",
    font_size: int = 250,
):
    """Render *word* with *gfont_metadata* and save into *output_dir*.

    Returns Path to the produced SVG (mainly for debugging / progress logging).
    """
    font_file_name: str = gfont_metadata["filename"]
    font_dir_name: str = gfont_metadata["font_dir_name"]

    if font_path is None:
        font_path = load_font(gfont_metadata, font_base_dir)

    buf = BytesIO()
    renderTextToObj(
        font_path,
        word,
        buf,
        fontSize=font_size,
        margin=0,
        backendName=backend,
    )
    buf.seek(0)
    text_svg = buf.read().decode("utf-8")

    # write svg
    output_hash, input_hash_text = get_font_file_name_text_hash(
        font_file_name, word, True
    )

    if output_dir is not None and save_svg:
        output_svg_dir = output_dir / "svg"
        output_svg_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_svg_dir / f"{output_hash}.svg"
        # Skip work if already rendered (idempotent, handy when resuming).
        if not output_path.exists():
            renderText(font_path, word, output_path)

    # write png
    if output_dir is not None and save_png:
        output_png_dir = output_dir / "png"
        output_png_dir.mkdir(exist_ok=True, parents=True)
        output_png_path = output_png_dir / f"{output_hash}.png"
        if not output_png_path.exists():
            renderText(font_path, word, output_png_path)

    # write json
    # font metadata specs: https://googlefonts.github.io/gf-guide/metadata.html
    font_name = gfont_metadata["name"]
    font_full_name = gfont_metadata["fullName"]
    group_tags = gfont_metadata["group_tags"]
    group_tag_weights = gfont_metadata["group_tag_weights"]
    source = gfont_metadata["source"]
    style = gfont_metadata["style"]
    weight = gfont_metadata["weight"]
    num_font_variants = gfont_metadata["num_font_variants"]
    category = gfont_metadata["category"]
    classifications = gfont_metadata.get("classifications", None)
    stroke = gfont_metadata.get("stroke", None)

    data = {
        "font_name": font_name,
        "font_full_name": font_full_name,
        "font_dir_name": font_dir_name,
        "font_file_name": font_file_name,
        "group_tags": group_tags,
        "group_tag_weights": group_tag_weights,
        "source": source,
        "word": word,
        "input_hash_text": input_hash_text,
        "output_hash": output_hash,
        "text_svg": text_svg,
        "style": style,
        "weight": weight,
        "num_font_variants": num_font_variants,
        "category": category,
        "classifications": classifications,
        "stroke": stroke,
    }
    return json.dumps(data)


def create_batched_args_iter(
    words,
    gfont_metadata_list,
    font_base_dir,
    args,
):
    batch_size = args.batch_size
    backend = args.backend
    font_size = args.font_size
    print(
        "Args for batched args iter:"
        f"batch_size: {batch_size}, "
        f"backend: {backend}, "
        f"font_size: {font_size}"
    )
    """Create batches of words per font to reduce font loading overhead"""
    for gfont_meta in gfont_metadata_list:
        for i in range(0, len(words), batch_size):
            word_batch = words[i : i + batch_size]
            yield word_batch, gfont_meta, font_base_dir, backend, font_size


def batch_worker(args):
    """Process a batch of words with the same font"""
    word_batch, gfont_meta, font_base_dir, backend, font_size = args
    font_path = load_font(gfont_meta, font_base_dir)
    results = []
    for word in word_batch:
        result = render_word_font(
            word, gfont_meta, font_path, backend=backend, font_size=font_size
        )
        results.append(result)
    return results


@click.command()
@click.option(
    "--input_words_path",
    default="data/processed/words_curation/pangrams.txt",
    help="Path to words file.",
)
@click.option(
    "--input_text",
    default=None,
    help="Path to words file.",
)
@click.option(
    "--input_gfont_metadata_jsonl_path",
    default="data/google_font_processor/google_font_metadata.filtered.jsonl",
    help="Path to words file.",
)
@click.option("--word_split_tsv", default=None, help="Path to words split file.")
@click.option("--font_split_tsv", default=None, help="Path to fonts split file.")
@click.option(
    "--num_workers",
    default=None,
    type=int,
    help="Number of parallel worker processes (default: number of CPUs).",
)
@click.option(
    "--output_dir",
    default="data/processed/svg_word_generation/pangrams",
    help="Path to output dir.",
)
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite existing output."
)
@click.option(
    "--chunk_size",
    default=50_000,
    type=int,
    help="Number of rows to save in each chunk.",
)
@click.option(
    "--batch_size",
    default=5000,
    type=int,
    help="Number of rows to save in each chunk.",
)
@click.option(
    "--backend",
    default="skia",
    type=click.Choice(["svg", "skia"]),
    help="Backend to use for rendering.",
)
@click.option("--font_size", default=250, type=int)
def main(**kwargs) -> None:
    args = SimpleNamespace(**kwargs)

    # prepare output dir
    output_dir = Path(getattr(args, "output_dir", None))
    overwrite = getattr(args, "overwrite", None)
    overwrite = True
    print(f"Must overwrite: {overwrite}. No resume support yet.")
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
            print(f"Output dir {output_dir} already exists. Overwriting.")
        else:
            print(f"Output dir {output_dir} already exists. continuing.")
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output dir: {output_dir}")

    # Prepare dataset
    input_text = getattr(args, "input_text", None)
    if input_text is not None:
        print(f"Reading input text: {input_text}. Skip reading words from files")
        words = [input_text]
    else:
        input_words_path = Path(getattr(args, "input_words_path", None))

        print(f"Reading words from {input_words_path}")
        words = read_words(input_words_path)

    gfont_metadata_jsonl_path = Path(
        getattr(args, "input_gfont_metadata_jsonl_path", None)
    )
    gfont_metadata_list = [
        json.loads(line) for line in gfont_metadata_jsonl_path.read_text().splitlines()
    ]

    # split word and font if needed
    word_split_tsv = getattr(args, "word_split_tsv", None)
    font_split_tsv = getattr(args, "font_split_tsv", None)

    if word_split_tsv is not None:
        print(f"Reading word split from {word_split_tsv}")
        word_split_df = pd.read_csv(word_split_tsv, sep="\t")
        word_index = word_split_df["index"].tolist()
        print(f"Number of word: {len(words)} -> {len(word_index)}")
        words = [words[i] for i in word_index]

    if font_split_tsv is not None:
        print(f"Reading font split from {font_split_tsv}")
        font_split_df = pd.read_csv(font_split_tsv, sep="\t")
        font_index = font_split_df["index"].tolist()
        print(f"Number of font: {len(gfont_metadata_list)} -> {len(font_index)}")
        gfont_metadata_list = [gfont_metadata_list[i] for i in font_index]

    # [NOTE](xk): load font locally, which is even faster.
    font_base_dir = Path("data/google_fonts/ofl")

    # test render one word
    word = words[0]
    gfont_meta = gfont_metadata_list[0]
    font_path = load_font(gfont_meta, font_base_dir)
    data = render_word_font(word, gfont_meta, font_path)

    # multiprocessing
    num_workers = getattr(args, "num_workers", None)
    if num_workers is None:
        cpu_count = os.cpu_count()
        if cpu_count is None:
            cpu_count = 8  # arbitrary default
        num_workers = min(20, cpu_count)  # good default for I/O
    print(f"Using {num_workers} workers")

    num_words = len(words)
    num_fonts = len(gfont_metadata_list)
    tasks_total = num_words * num_fonts
    print(f"Rendering {tasks_total} tasks (#words {num_words}, #fonts {num_fonts})...")

    chunk_size = getattr(args, "chunk_size", None)
    chunk_buffer = []
    output_metadata_dir = output_dir / "metadata"
    output_metadata_dir.mkdir(exist_ok=True, parents=True)
    chunk_idx = 0

    backend = getattr(args, "backend", "skia")
    print(f"Using backend: {backend}")
    pbar = tqdm.tqdm(total=tasks_total, desc="Rendering")
    batch_size = getattr(args, "batch_size", 5000)
    with multiprocessing.Pool(processes=num_workers) as pool:
        chunk_buffer = []

        args_iter = create_batched_args_iter(
            words,
            gfont_metadata_list,
            font_base_dir,
            args,
        )

        for result in pool.imap_unordered(batch_worker, args_iter):
            chunk_buffer.extend(result)
            pbar.update(len(result))

            # write chunk
            if len(chunk_buffer) >= chunk_size:
                output_metadata_path = output_metadata_dir / f"{chunk_idx:05d}.jsonl"
                pbar.write(
                    f"Writing {len(chunk_buffer)} rows to {output_metadata_path}"
                )
                chunk_idx += 1

                with open(output_metadata_path, "a", buffering=8192 * 16 * 16) as f:
                    for data in chunk_buffer:
                        f.write(data + "\n")
                chunk_buffer.clear()

        # write final chunk
        if chunk_buffer:
            output_metadata_path = output_metadata_dir / f"{chunk_idx:05d}.jsonl"
            pbar.write(f"Writing {len(chunk_buffer)} rows to {output_metadata_path}")
            with open(output_metadata_path, "a", buffering=8192 * 16) as f:
                for data in chunk_buffer:
                    f.write(data + "\n")
            chunk_buffer.clear()

    print(f"Done. SVGs are in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
