#!/usr/bin/env python3
"""
python scripts/tools/download_model_from_storage.py -i workspace/hf_downloads/Qwen/Qwen3-1.7B -o saves/Qwen/Qwen3-1.7B

model_path="saves/Qwen/Qwen3-1.7B"
python -m sglang.launch_server \
    --host "localhost" \
    --model-path "${model_path}" \
    --port 30000 \
    --tp 1 \
    --dp 1 \
    --log-level debug

# Use --log-requests to make sure the sampling params are correct.

model_path="saves/Qwen/Qwen3-1.7B"
data=/home/vecglypher/mnt/workspace/svg_glyph_llm/data/processed/filtered_sft/250903-alphanumeric/ood_font_family_decon
output_dir=outputs/debug/
python src/serve/api_infer.py \
    --data "${data}" \
    --output_dir "${output_dir}"/infer \
    --model "${model_path}" \
    --base_url http://localhost:30000/v1 \
    --max_samples 10 --debug

python -m src.serve.decode_to_svg "${output_dir}"/infer
python -m src.eval.svg2img_dir "${output_dir}"/infer_decoded "${output_dir}"/infer_decoded-img_base64-predict --field predict --width 192 --height 192
python -m src.eval.svg2img_dir "${output_dir}"/infer_decoded "${output_dir}"/infer_decoded-img_base64-label --field label --width 192 --height 192
python -m src.eval.build_eval_data --input_infer_jsonl_dir "${output_dir}"/infer_decoded --input_infer_img_base64_dir "${output_dir}"/infer_decoded-img_base64-predict --field predict

model_path="saves/Qwen/Qwen2.5-VL-7B-Instruct"
python -m sglang.launch_server \
    --host "localhost" \
    --model-path "${model_path}" \
    --port 30000 \
    --tp 1 \
    --dp 2

model_path="saves/Qwen/Qwen2.5-VL-7B-Instruct"
data=outputs/debug/ocr_eval_data
output_dir=outputs/debug/
python src/serve/api_infer.py \
    --data "${data}" \
    --output_dir "${output_dir}"/ocr_infer \
    --model "${model_path}" \
    --base_url http://localhost:30000/v1

python -m src.eval.score_ocr_eval "${output_dir}"/ocr_infer "${output_dir}"/eval_ocr_infer --no_use_case
python -m src.eval.score_ocr_eval "${output_dir}"/ocr_infer "${output_dir}"/eval_ocr_infer --use_case
"""
import asyncio
import base64
import hashlib
import io
import json
import logging
import mimetypes
import os
import random
import signal
import time
from pathlib import Path
from typing import Any, Dict, List

import click
import httpx
import openai
import tqdm
from omegaconf import OmegaConf
from openai import AsyncOpenAI
from PIL import Image

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@click.command()
# dataset params
@click.option("--data", default=None, help="Path to input JSONL.")
@click.option("--max_samples", type=int, default=None, help="Max samples to process.")
# output params
@click.option("--output_dir", default=None, help="Where to write *.jsonl shards.")
# async request params
@click.option("--concurrency", type=int, default=256, show_default=True)
@click.option(
    "--request_timeout",
    type=float,
    default=300.0,
    show_default=True,
    help="Per-request hard timeout (sec).",
)
# model params
@click.option(
    "--model",
    default=None,
    help="Model name as seen by the server (e.g., 'qwen2.5-7b-instruct').",
)
@click.option(
    "--base_url",
    default=os.getenv("OPENAI_BASE_URL", "http://localhost:30000/v1"),
    show_default=True,
    help="OpenAI-compatible base URL (default from OPENAI_BASE_URL).",
)
# sampling params
@click.option("--temperature", type=float, default=1.0, show_default=True)
@click.option("--top_p", type=float, default=1.0, show_default=True)
@click.option("--max_tokens", type=int, default=16, show_default=True)
@click.option(
    "--extra_body",
    default=r'{"chat_template_kwargs": {"enable_thinking": false}, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0}',
    # Qwen suggest using temperature=0.7, top_p=0.8, top_k=20, min_p=0, repetition_penalty=1.05.
    help="Extra body to pass to the API.",
    show_default=True,
    type=str,
)
# saving params
@click.option(
    "--resume/--no_resume",
    default=True,
    show_default=True,
    help="Resume from existing shards.",
)
@click.option(
    "--shard_size",
    type=int,
    default=1000,
    show_default=True,
    help="Lines per output shard.",
)
# addtinoal input args
@click.option("--input_args_path", type=str, default=None, help="Path to input args.")
@click.option("--debug", is_flag=True, default=False)
@click.option("--reasoning_effort", type=str, default=None)
def main(**kwargs):
    print("kwargs", kwargs)
    asyncio.run(run(kwargs))


def is_base64(s: str) -> bool:
    try:
        # Base64 strings must have length multiple of 4
        if len(s) % 4 != 0:
            return False

        # Try decoding
        decoded = base64.b64decode(s, validate=True)

        # Optional: re-encode and check if it matches (to avoid false positives)
        return base64.b64encode(decoded).decode("utf-8") == s
    except Exception:
        return False


def load_and_resize_to_data_url(path: str, max_side=None) -> str:
    """Resize long side to <= max_side (keeps aspect), encode as base64 data URL."""
    if is_base64(path):
        return f"data:image;base64,{path}"

    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"

    with Image.open(path) as im:
        im = im.convert("RGB")
        if max_side is not None:
            w, h = im.size
            scale = min(1.0, max_side / max(w, h))
            if scale < 1.0:
                im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        # Use JPEG for photos; PNG for line art/diagrams if you prefer.
        if mime == "image/png":
            im.save(buf, format="PNG", optimize=True)
        else:
            im.save(
                buf, format="JPEG", quality=85, optimize=True
            )  # good size/quality tradeoff
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return f"data:image;base64,{b64}"


# -------------------------------
# Utilities
# -------------------------------
def sha1_of_obj(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_processed_ids(out_dir: Path) -> set:
    """
    Scan existing *.jsonl shards for 'sample_id' (preferred) or 'input_hash' and
    return a set for skipping on resume.
    """
    done = set()
    for fp in sorted(out_dir.glob("*.jsonl")):
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        sid = obj.get("sample_id") or obj.get("input_hash")
                        if sid:
                            done.add(sid)
                    except Exception:
                        continue
        except FileNotFoundError:
            continue
    return done


def build_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Best-effort adapter for common dataset shapes:
      1) {"messages": [{"role": "user"/"system"/"assistant", "content": "..."}]}
      2) {"conversations": [{"role" or "from": "user"/"human"/"assistant"/"gpt"/"system", "content": "..."}]}
      3) {"instruction": "...", "input": "...", "system": "..."} (Alpaca-style)
    """
    if "messages" in sample and isinstance(sample["messages"], list):
        return sample["messages"]

    if "conversations" in sample and isinstance(sample["conversations"], list):
        raise NotImplemented
        msgs = []
        for turn in sample["conversations"]:
            role = turn.get("role")
            if role is None and "from" in turn:
                fr = turn["from"].lower()
                if fr in ("human", "user"):
                    role = "user"
                elif fr in ("assistant", "gpt", "bot"):
                    role = "assistant"
                elif fr == "system":
                    role = "system"
                else:
                    role = "user"
            content = turn.get("content", "")
            msgs.append({"role": role, "content": content})
        return msgs

    if "instruction" in sample:
        msgs = []

        sysmsg = sample.get("system")
        if sysmsg:
            msgs.append({"role": "system", "content": sysmsg})

        user_content = []
        images = sample.get("images", [])
        for img in images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": load_and_resize_to_data_url(img),
                    },
                },
            )

        user_content.append(
            {"type": "text", "text": sample["instruction"]},
        )
        # if sample.get("input"):
        #     user_content = f"{user_content}\n\nInput:\n{sample['input']}"
        msgs.append({"role": "user", "content": user_content})

        return msgs
    else:
        raise ValueError(f"Unsupported sample format: {sample}")

    # Fallback: treat entire sample as a single user prompt.
    return [{"role": "user", "content": json.dumps(sample, ensure_ascii=False)}]


def make_sample_id(sample: Dict[str, Any], idx: int) -> str:
    # Prefer explicit id fields if present
    for key in ("id", "sample_id", "uid", "guid"):
        if key in sample:
            return str(sample[key])
    # Otherwise hash content + index for stability
    return f"{idx}:{sha1_of_obj(sample)[:16]}"


# -------------------------------
# Chunked writer (single consumer)
# -------------------------------
try:
    import orjson as _orjson

    _dumps = lambda o: _orjson.dumps(o).decode("utf-8")
except Exception:
    _dumps = lambda o: json.dumps(o, ensure_ascii=False)


class ChunkedJSONLWriter:
    def __init__(
        self,
        out_dir: Path,
        max_lines_per_shard: int = 2000,
        prefix: str = "preds",
        total=None,
        *,
        flush_every_lines: int = 64,  # batch size before flushing
        flush_every_seconds: float = 2.0,  # also flush on time (helps when batches are small)
        fsync: bool = False,  # fsync is OFF by default (massive speedup)
        fsync_interval_seconds: float = 30.0,  # if fsync=True, do it at most this often
        open_buffer_bytes: int = 4 << 20,  # 4 MiB buffered writes
    ):
        self.out_dir = out_dir
        ensure_dir(out_dir)
        self.max_lines = max_lines_per_shard
        self.prefix = prefix

        self._fp = None
        self._lines = 0
        self._shard_idx = 0

        self._buf: List[str] = []
        self._last_flush = time.monotonic()
        self._last_fsync = time.monotonic()

        self.flush_every_lines = flush_every_lines
        self.flush_every_seconds = flush_every_seconds
        self.fsync_enabled = fsync
        self.fsync_interval_seconds = fsync_interval_seconds
        self.open_buffer_bytes = open_buffer_bytes

        if total:
            self.pbar = tqdm.tqdm(total=total, desc="Write")
        else:
            err_msg = "total must be specified, to sync the already written samples"
            logger.error(err_msg)
            raise ValueError(err_msg)

    def _open_new(self):
        if self._fp:
            self._flush(force=True)
            self._fp.close()

        # resume logic: skip full shards; resume partial
        existed_lines = 0
        while True:
            fname = f"{self.prefix}-{self._shard_idx:05d}.jsonl"
            fpath = self.out_dir / fname
            if not fpath.exists():
                break

            existed_lines = count_lines(fpath)
            if existed_lines > self.max_lines:
                raise ValueError(
                    f"Shard {fpath} has too many lines: {existed_lines} > {self.max_lines}"
                )
            elif existed_lines == self.max_lines:
                logger.info(
                    f"Shard {fpath} is full with {existed_lines} lines, skipping"
                )
                self.pbar.update(existed_lines)
                self._shard_idx += 1
            else:
                logger.info(
                    f"Shard {fpath} is partially written with {existed_lines} lines, resuming"
                )
                self.pbar.update(existed_lines)
                break

        # big buffered append
        self._fp = open(fpath, "a", encoding="utf-8", buffering=self.open_buffer_bytes)
        self._lines = existed_lines
        self._shard_idx += 1
        self._buf.clear()
        self._last_flush = time.monotonic()
        self._last_fsync = self._last_flush

    def _flush(self, *, force: bool = False):
        if not self._fp or not self._buf:
            return
        self._fp.writelines(self._buf)
        self._buf.clear()
        self._fp.flush()

        # optional durability: sync at most every N seconds, or when forced
        if self.fsync_enabled:
            now = time.monotonic()
            if force or (now - self._last_fsync) >= self.fsync_interval_seconds:
                os.fsync(self._fp.fileno())
                self._last_fsync = now

        self._last_flush = time.monotonic()

    def write(self, obj: Dict[str, Any] | str):
        if self._fp is None or self._lines >= self.max_lines:
            self._open_new()

        # accept dicts or pre-serialized strings/lines
        if isinstance(obj, str):
            line = obj if obj.endswith("\n") else obj + "\n"
        else:
            line = _dumps(obj) + "\n"

        self._buf.append(line)
        self._lines += 1

        if self.pbar:
            self.pbar.update(1)

        # flush on batch size or time
        if (
            len(self._buf) >= self.flush_every_lines
            or (time.monotonic() - self._last_flush) >= self.flush_every_seconds
        ):
            self._flush()

    def close(self):
        if self._fp:
            self._flush(force=True)  # make sure last batch hits disk
            self._fp.close()
            self._fp = None


# -------------------------------
# Retry helper
# -------------------------------
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


async def with_retries(
    coro_factory, *, max_attempts=8, base_delay=1.0, max_delay=20.0, jitter=True
):
    attempt = 0
    while True:
        try:
            return await coro_factory()
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            pass  # retry
        except openai.APIStatusError as e:
            code = getattr(e, "status_code", None)
            if code not in RETRYABLE_STATUS:
                raise RuntimeError(f"Non retryable API code {code}: {e}")
        except httpx.TransportError:
            pass  # network hiccup

        attempt += 1
        if attempt >= max_attempts:
            raise RuntimeError(f"Failed after {max_attempts} attempts")
        delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
        if jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        await asyncio.sleep(delay)


# -------------------------------
# Worker and writer coroutines
# -------------------------------
async def worker(
    name: str,
    client: AsyncOpenAI,
    model: str,
    in_queue: "asyncio.Queue[Dict[str, Any]]",
    out_queue: "asyncio.Queue[Dict[str, Any]]",
    request_timeout: float,
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: Dict[str, Any],
    reasoning_effort,
):
    while True:
        item = await in_queue.get()
        if item is None:
            in_queue.task_done()
            break

        sample = item["sample"]
        sample_id = item["sample_id"]
        input_hash = item["input_hash"]

        # NOTE: label/output field name for llama-factory
        label = sample.get("output", None)
        metadata = sample.get("metadata", None)

        messages = build_messages(sample)

        async def call():
            # Hard timeout guard around the API call
            return await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    reasoning_effort=reasoning_effort,
                ),
                timeout=request_timeout,
            )

        result_obj: Dict[str, Any]
        t0 = time.time()
        try:
            completion = await with_retries(call)
            latency = time.time() - t0
            text = completion.choices[0].message.content if completion.choices else ""
            usage = getattr(completion, "usage", None)
            result_obj = {
                "sample_id": sample_id,
                "input_hash": input_hash,
                # output fields
                "predict": text,
                "label": label,
                "metadata": metadata,
                # other fields
                "usage": usage.to_dict() if usage else None,
                "latency_sec": round(latency, 3),
                # "raw": completion.to_dict(),  # keep for auditing; remove if too large
            }
            await out_queue.put(result_obj)
        except Exception as e:
            latency = time.time() - t0
            result_obj = {
                "sample_id": sample_id,
                "input_hash": input_hash,
                "error": f"{type(e).__name__}: {e}",
                "latency_sec": round(latency, 3),
            }
            logger.error(f"{name} failed: {result_obj}")

        in_queue.task_done()


async def writer(
    out_dir: Path,
    out_queue: "asyncio.Queue[Dict[str, Any]]",
    max_lines_per_shard: int,
    total=None,
):
    # Move blocking file I/O off the event loop.
    w = ChunkedJSONLWriter(
        out_dir,
        max_lines_per_shard,
        total=total,
        flush_every_lines=64,
        flush_every_seconds=2.0,
        fsync=False,  # <-- leave off unless you truly need durability mid-run
        fsync_interval_seconds=30.0,
        open_buffer_bytes=4 << 20,
    )
    try:
        while True:
            obj = await out_queue.get()
            if obj is None:
                out_queue.task_done()
                break
            # run write in a thread to avoid blocking the loop on slow disks
            await asyncio.to_thread(w.write, obj)
            out_queue.task_done()
    finally:
        await asyncio.to_thread(w.close)


# -------------------------------
# Main driver
# -------------------------------
async def run(args):
    # Parse args or load from file
    if args.get("input_args_path"):
        args = OmegaConf.load(args.get("input_args_path"))
    else:
        args = OmegaConf.create(args)
    for check_key in ["data", "output_dir", "model"]:
        if not args.get(check_key):
            raise ValueError(f"Missing required argument: {check_key}")

    ensure_dir(Path(args.output_dir))
    # Save args
    output_args_path = Path(args.output_dir) / "args.yaml"
    with open(output_args_path, "w") as f:
        f.write(OmegaConf.to_yaml(args))
    logger.info(f"Output args saved to {output_args_path}")

    # Setup logging
    output_log_path = Path(args.output_dir) / "log.txt"
    handler = logging.FileHandler(output_log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    logger.info(f"Running with args:\n{OmegaConf.to_yaml(args)}")

    # OpenAI-compatible client
    client = AsyncOpenAI(
        base_url=args.base_url,  # e.g., http://localhost:8000/v1
        api_key=os.environ.get(
            "OPENAI_API_KEY", "EMPTY"
        ),  # vLLM/FastChat can accept arbitrary key
    )

    # Test run one sample
    sample = next(load_jsonl(args.data, decode_jsonl=True, logger=logger))
    messages = build_messages(sample)

    try:
        extra_body = json.loads(args.extra_body)
    except json.decoder.JSONDecodeError as e:
        err_msg = f"Invalid extra_body: {args.extra_body}: {e}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    if args.debug:
        logger.info("start debugging...")
    output = await asyncio.wait_for(
        client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            extra_body=extra_body,
            reasoning_effort=args.reasoning_effort,
        ),
        timeout=args.request_timeout,
    )
    if args.debug:
        logger.info(f"Test run: {output}")
        # fmt: off
        breakpoint()
        # fmt: on
        return

    # Resume set
    already = load_processed_ids(Path(args.output_dir)) if args.resume else set()
    logger.info(f"Loaded {len(already)} processed ids from {args.output_dir}")

    if not args.resume:
        # delete all *.jsonl in output_dir
        for fp in Path(args.output_dir).glob("*.jsonl"):
            fp.unlink()
        logger.info(f"Deleted all *.jsonl in {args.output_dir}")

    num_lines = count_lines(args.data)
    logger.info(f"Found {num_lines} lines in {args.data}.")
    max_samples = min(args.max_samples, num_lines) if args.max_samples else num_lines
    logger.info(f"Will process {max_samples} samples.")

    in_q: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    out_q: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)

    # Start writer
    writer_task = asyncio.create_task(
        writer(Path(args.output_dir), out_q, args.shard_size, total=max_samples)
    )

    # Start workers
    extra_body = json.loads(args.extra_body)
    workers = [
        asyncio.create_task(
            worker(
                name=f"w{i}",
                client=client,
                model=args.model,
                in_queue=in_q,
                out_queue=out_q,
                request_timeout=args.request_timeout,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                extra_body=extra_body,
                reasoning_effort=args.reasoning_effort,
            )
        )
        for i in range(args.concurrency)
    ]

    # Graceful shutdown
    stop = asyncio.Event()

    def _graceful_shutdown(*_):
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _graceful_shutdown)
        except NotImplementedError:
            pass

    # Producer: stream the dataset
    total_enqueued = 0
    total_skipped = 0
    # pbar = tqdm.tqdm(total=max_samples, desc="Enqueue")
    for idx, line in enumerate(
        load_jsonl(args.data, decode_jsonl=False, logger=logger)
    ):
        logger.debug(
            f"Enqueued {idx} lines, total enqueued {total_enqueued}, total skipped {total_skipped}"
        )
        if stop.is_set():
            break

        if total_enqueued + total_skipped >= max_samples:
            break

        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
        except Exception:
            # keep a record of malformed lines
            sid = f"malformed:{idx}"
            result_obj = {
                "sample_id": sid,
                "input_hash": None,
                "error": "JSONDecodeError",
                "raw_line": line[:5000],
            }
            logger.error(f"Failed to parse line {idx}: {result_obj}")
            continue

        sample_id = make_sample_id(sample, idx)
        input_hash = sha1_of_obj(sample)

        if args.resume and (sample_id in already or input_hash in already):
            logger.debug(f"Skipping {sample_id} (already processed)")
            total_skipped += 1
        else:
            await in_q.put(
                {
                    "sample": sample,
                    "sample_id": sample_id,
                    "input_hash": input_hash,
                }
            )
            total_enqueued += 1
        # pbar.update(1)

    # Drain
    for _ in workers:
        await in_q.put(None)
    await in_q.join()

    # Tell writer to finish
    await out_q.put(None)
    await out_q.join()

    # Await workers & writer
    for t in workers:
        await t
    await writer_task

    await client.close()

    total_input_processed = total_enqueued + total_skipped
    logger.info(
        f"Enqueued {total_enqueued} new samples. Skip {total_skipped} old samples."
    )
    total_output_processed = count_lines(args.output_dir)

    incomplete_file = Path(args.output_dir) / "incomplete.txt"
    complete_file = Path(args.output_dir) / "complete.txt"

    incomplete_file.unlink(missing_ok=True)
    complete_file.unlink(missing_ok=True)

    if total_output_processed != max_samples:
        logger.error(
            f"[Imcomplete] Expected to process {max_samples} samples, but only processed {total_output_processed}."
        )
        incomplete_file.write_text("")
    else:
        logger.info(f"[Complete] Processed {total_output_processed} samples.")
        complete_file.write_text("")


def _count_lines(filename):
    count = 0
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            count += chunk.count(b"\n")
    return count


def count_lines(filename_or_dir):
    filename_or_dir = Path(filename_or_dir)
    if not filename_or_dir.exists():
        raise FileNotFoundError(f"Path not found: {filename_or_dir}")

    if os.path.isfile(filename_or_dir):
        return _count_lines(filename_or_dir)
    else:
        return sum(_count_lines(f) for f in Path(filename_or_dir).glob("*.jsonl"))


def _load_jsonl(jsonl_path, decode_jsonl=True):
    with open(jsonl_path, "r") as f:
        for line in f:
            yield json.loads(line.strip()) if decode_jsonl else line.strip()


def load_jsonl(jsonl_path_or_dir, logger=None, decode_jsonl=True):
    jsonl_path_or_dir = Path(jsonl_path_or_dir)
    if not jsonl_path_or_dir.exists():
        raise FileNotFoundError(f"Path not found: {jsonl_path_or_dir}")

    if jsonl_path_or_dir.is_file():
        yield from _load_jsonl(jsonl_path_or_dir, decode_jsonl)
        return

    jsonl_files = sorted(jsonl_path_or_dir.glob("*.jsonl"))
    if logger:
        logger.info(f"Found {len(jsonl_files)} jsonl files from {jsonl_path_or_dir}.")

    count = 0
    for jsonl_file in jsonl_files:
        if logger:
            logger.info(f"Loading {jsonl_file}...")
        for record in _load_jsonl(jsonl_file, decode_jsonl):
            count += 1
            yield record

    if logger:
        logger.info(f"Loaded {count} records from {jsonl_path_or_dir}.")


if __name__ == "__main__":
    main()
