#!/usr/bin/env python3
"""
svg2img_dir.py
Convert SVG text stored in JSONL files (field: "predict") to images with CairoSVG.
Parallel rendering with ProcessPoolExecutor.
Finally, we pack all images into base64 chunk files.

Usage:
    python -m src.eval.svg2img_dir INPUT_DIR OUTPUT_DIR --field predict
        --workers 8] [--width 1024] [--height 1024] [--dpi 300] [--scale 1.0]
        [--background white] [--overwrite] [--verbose]

Notes:
- Output defaults to INPUT_DIR / "render".
- Each input JSONL gets its own subfolder inside the output dir.
- Filenames are <jsonl_stem>_<line_number>.png
- The final outputs are chunked base64 files. The format:
    {"img_hash": img_hash,
     "img_base64": img_base64}
"""
import base64
import io
import json
import os
import shutil
import signal
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import cairosvg
import click
import cv2
import numpy as np
import tqdm
from PIL import Image

from ..serve.api_infer import count_lines
from ..svg_glyph_gen_v2.filter_by_pangram_svg import blake2_hash
from ..svg_glyph_gen_v2.utils import (
    count_lines,
    load_jsonl_by_generator,
    prepare_output_dir_and_logger,
    write_jsonl,
)

TIMEOUT_SECS = 3  # per-task max runtime


def _run_with_timeout(fn, arg, timeout=TIMEOUT_SECS):
    """Run `fn(arg)` in a worker with a hard timeout using SIGALRM."""

    def _raise_timeout(signum, frame):
        raise TimeoutError("timed out")

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return fn(arg)
    except TimeoutError as e:
        # Best-effort name extraction (adjust to your payload shape)
        name = (
            getattr(arg, "name", None)
            or (arg.get("name") if isinstance(arg, dict) else None)
            or "<unknown>"
        )
        print(f"Timed out after {timeout}s: {name}: arg {arg}")
        # return ("err", arg, f"Timed out after {timeout}s")
        # XXX: in case of timeout, find those svg leads to png saving timeout
        raise e
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _timed_render_task(payload):
    # call your original worker under the timeout
    return _run_with_timeout(_render_task, payload, timeout=TIMEOUT_SECS)


def _render_task(payload):
    """
    Worker process: render one SVG string to a PNG path.
    Returns (status, name, message) where status in {'ok','skip','err'}.
    """
    (
        svg_text,
        out_path_str,
        overwrite,
        output_width,
        output_height,
        dpi,
        scale,
        background_color,
        base_url,
    ) = payload

    out_path = Path(out_path_str)
    try:
        if out_path.exists():
            # NOTE: Even if out_path exists, indexed_softlink_output_path may not exist
            # link_softlink(out_path, indexed_softlink_output_path)
            if not overwrite:
                return ("skip", out_path, None)
            else:
                out_path.unlink()

        out_path.parent.mkdir(parents=True, exist_ok=True)

        cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            write_to=str(out_path),
            output_width=output_width,
            output_height=output_height,
            dpi=dpi,
            scale=scale,
            background_color=background_color,
            url=base_url,
        )

        if not out_path.exists():
            raise RuntimeError(f"Failed to write {out_path}")

        # link_softlink(out_path, indexed_softlink_output_path)
        return ("ok", out_path, None)
    except Exception as e:
        # write a white image to path
        if output_width is None:
            output_width = output_height
        img = np.full((output_height, output_width, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(out_path), img)

        # link_softlink(out_path, indexed_softlink_output_path)
        return ("err", out_path, str(e))


def link_softlink(out_path, indexed_softlink_output_path):
    if indexed_softlink_output_path is not None:
        if indexed_softlink_output_path.is_symlink():
            indexed_softlink_output_path.unlink()
        elif indexed_softlink_output_path.exists():
            raise FileExistsError(
                f"{indexed_softlink_output_path} exists and isn’t a symlink."
            )
        rel = os.path.relpath(out_path, start=indexed_softlink_output_path.parent)
        indexed_softlink_output_path.symlink_to(rel)
    else:
        raise ValueError("indexed_softlink_output_path is None")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument(
    "input_dir",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    # default="outputs/debug/infer_decoded",
    required=True,
)
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--field",
    default="predict",
    show_default=True,
    help="JSON field containing SVG text.",
)
@click.option(
    "--width", "output_width", type=int, default=None, help="Output width in pixels."
)
@click.option(
    "--height", "output_height", type=int, default=128, help="Output height in pixels."
)
@click.option("--dpi", type=int, default=96, help="Render DPI.")
@click.option(
    "--scale",
    type=float,
    default=1,
    help="Scale factor (alternative to width/height).",
)
@click.option(
    "--background",
    "background_color",
    default="white",
    help="Background color, e.g. 'white' or '#FFFFFF'.",
)
@click.option(
    "--workers",
    type=int,
    default=20,
    show_default=True,
    help="Number of parallel worker processes.",
)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, help="Print progress and errors.")
@click.option("--debug", is_flag=True, help="Print debug info.")
def main(
    input_dir: Path,
    output_dir: Path,
    field: str,
    output_width: Optional[int],
    output_height: Optional[int],
    dpi: Optional[int],
    scale: Optional[float],
    background_color: Optional[str],
    workers: int,
    overwrite: bool,
    verbose: bool,
    debug: bool,
) -> None:
    """Convert all *.jsonl in INPUT_DIR where each line has SVG text in FIELD into PNGs."""
    _, logger = prepare_output_dir_and_logger(
        output_dir=output_dir,
        overwrite=False,
    )
    # NOTE: overwrite is handled internally by _render_task
    logger.warning("Overwrite is handled internally by _render_task")

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.info("No *.jsonl files found.")
        raise SystemExit(1)

    # Counters
    total_ok = 0
    total_skip_exist = 0
    total_err_render = 0
    total_skip_input = 0  # empty lines or missing/empty field
    total_err_json = 0  # bad JSON

    temp_output_dir = tempfile.mkdtemp(prefix=f"{input_dir.name}")
    temp_output_dir = Path(temp_output_dir)
    logger.info(f"temp_output_dir: {temp_output_dir}")

    def gen_payloads():
        """Yield payloads for valid lines; count input-side skips/errors."""
        nonlocal total_skip_input, total_err_json
        for jf in jsonl_files:
            if verbose:
                logger.info(f"[.] Scanning {jf.name}")

            out_dir = temp_output_dir
            base_url = jf.parent.resolve().as_uri() + "/"

            with jf.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):

                    obj = json.loads(line)
                    svg_text = obj.get(field, None)
                    if svg_text is None:
                        raise ValueError(
                            f"Missing field {field} in {obj.keys()}\n{jf}:{idx}"
                        )

                    out_path = out_dir / f"{blake2_hash(svg_text)}.png"
                    yield (
                        svg_text,
                        str(out_path),
                        overwrite,
                        output_width,
                        output_height,
                        dpi,
                        scale,
                        background_color,
                        base_url,
                    )

    # test one file
    for inputs in gen_payloads():
        status, name, msg = _render_task(inputs)
        logger.info(f"test loading: {status}, {name}, {msg}")
        break
    if debug:
        # fmt: off
        from IPython import embed; embed()
        # fmt: on

    total_samples = count_lines(input_dir)
    pbar = tqdm.tqdm(total=total_samples, desc="JSONL files")
    if workers == 0:
        for payload in gen_payloads():
            status, name, msg = _render_task(payload)
            if status == "ok":
                total_ok += 1
                if verbose:
                    logger.info(f"    + Wrote {name}")
            elif status == "skip":
                total_skip_exist += 1
                if verbose:
                    logger.info(f"    - Skip (exists): {name}")
            else:  # "err" (includes timeout)
                total_err_render += 1
            pbar.update(1)
    else:
        # Render in parallel
        with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
            for status, name, msg in ex.map(
                _timed_render_task, gen_payloads(), chunksize=1
            ):
                if status == "ok":
                    total_ok += 1
                    if verbose:
                        logger.info(f"    + Wrote {name}")
                elif status == "skip":
                    total_skip_exist += 1
                    if verbose:
                        logger.info(f"    - Skip (exists): {name}")
                else:  # "err" (includes timeout)
                    total_err_render += 1
                    logger.info(f"    ! Render error: {name}: {msg}")
                pbar.update(1)

    logger.info(
        f"Done. ok={total_ok}, skipped_input={total_skip_input}, "
        f"skipped_exist={total_skip_exist}, json_errors={total_err_json}, render_errors={total_err_render}"
    )
    logger.info(f"Output dir: {output_dir}")

    pack_imgs_to_chunk(
        input_dir=input_dir,
        field=field,
        output_dir=output_dir,
        temp_output_dir=temp_output_dir,
        logger=logger,
        max_workers=workers,
    )

    # clean up temp_output_dir
    logger.info(f"Cleaning up temp_output_dir: {temp_output_dir}")
    shutil.rmtree(temp_output_dir)
    logger.info(f"Done cleaning up temp_output_dir: {temp_output_dir}")


def pack_imgs_to_chunk(
    input_dir,
    field,
    output_dir,
    temp_output_dir,
    logger,
    max_workers: int = 8,
):
    # Prepare jobs first (no heavy work here)
    num_samples = count_lines(input_dir)

    def get_job_iter():
        for data in load_jsonl_by_generator(input_dir):
            svg_text = data.get(field)
            if svg_text is None:
                raise ValueError(f"Missing field {field} in {data.keys()}")
            img_hash = blake2_hash(svg_text)
            img_path = Path(temp_output_dir) / f"{img_hash}.png"
            if not img_path.exists():
                raise ValueError(
                    f"img_path {img_path} does not exist (This should not happen)."
                )
            yield (img_path, img_hash)

    logger.info("start packing imgs to chunk")
    result_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        ex_map_iter = ex.map(
            load_pil_img_to_base64, (img_path for img_path, _ in get_job_iter())
        )
        pbar = tqdm.tqdm(total=num_samples, desc="Packing imgs to chunk")
        for (_, img_hash), img_base64 in zip(get_job_iter(), ex_map_iter):
            rec = {
                "img_hash": img_hash,
                "img_base64": img_base64,
            }
            result_list.append(rec)
            pbar.update(1)
    logger.info(f"Done. {len(result_list)} records")

    output_path = Path(output_dir) / "base64_image.jsonl"
    write_jsonl(result_list, output_path, logger=logger)


def load_pil_img_to_base64(img_path):
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64


if __name__ == "__main__":
    main()
