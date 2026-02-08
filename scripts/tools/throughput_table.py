#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODELS = [
    {
        "key": "gemma3-4b",
        "name": "Gemma3-4B",
        "path": "saves/google/gemma-3-4b-it",
        "tp": 1,
        "dp": 1,
        "gpus": "0",
    },
    {
        "key": "gemma3-27b",
        "name": "Gemma3-27B",
        "path": "saves/google/gemma-3-27b-it",
        "tp": 1,
        "dp": 1,
        "gpus": "0",
    },
    {
        "key": "llama3.3-70b",
        "name": "Llama3.3-70B",
        "path": "saves/meta-llama/Llama-3.3-70B-Instruct",
        "tp": 2,
        "dp": 1,
        "gpus": "0,1",
    },
]


def iter_jsonl(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def extract_text(sample):
    meta = sample.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = None
    if isinstance(meta, dict):
        text = meta.get("content_str")
        if text is not None:
            return text
    instruction = sample.get("instruction") or ""
    for line in instruction.splitlines():
        if line.startswith("Text content:"):
            return line.split("Text content:", 1)[1].strip()
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Run vLLM throughput benchmark and write a speed table."
    )
    parser.add_argument(
        "--data",
        default="data/processed/filtered_sft/250903-alphanumeric/ood_font_family_decon",
    )
    parser.add_argument("--output_dir", default="misc/throughput")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--model",
        action="append",
        help="Model key to run (gemma3-4b, gemma3-27b, llama3.3-70b).",
    )
    parser.add_argument("--skip_infer", action="store_true")
    args = parser.parse_args()

    models = MODELS
    if args.model:
        keys = {k.lower() for k in args.model}
        models = [m for m in MODELS if m["key"] in keys]
        if not models:
            raise SystemExit(f"No models matched: {sorted(keys)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = []

    for model in models:
        model_dir = output_dir / model["key"]
        infer_dir = model_dir / "infer"
        model_dir.mkdir(parents=True, exist_ok=True)

        run_meta_path = model_dir / "run_meta.json"
        if not args.skip_infer:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = model["gpus"]
            env["VLLM_HOST_IP"] = "localhost"
            env["VLLM_LOOPBACK_IP"] = "localhost"

            launch_cmd = [
                sys.executable,
                "src/serve/launch_server.py",
                model["path"],
                "--host",
                "localhost",
                "--port",
                str(args.port),
                "--data-parallel-address",
                "localhost",
                "-dp",
                str(model["dp"]),
                "-tp",
                str(model["tp"]),
            ]
            subprocess.run(launch_cmd, check=True, env=env)

            infer_cmd = [
                sys.executable,
                "src/serve/api_infer.py",
                "--data",
                args.data,
                "--output_dir",
                str(infer_dir),
                "--model",
                model["path"],
                "--base_url",
                f"http://localhost:{args.port}/v1",
                "--max_tokens",
                str(args.max_tokens),
                "--temperature",
                str(args.temperature),
                "--top_p",
                str(args.top_p),
                "--concurrency",
                str(args.concurrency),
                "--no_resume",
            ]
            if args.max_samples > 0:
                infer_cmd += ["--max_samples", str(args.max_samples)]

            t0 = time.time()
            try:
                subprocess.run(infer_cmd, check=True, env=env)
            finally:
                subprocess.run(
                    ["bash", "scripts/tools/slow_safe_pkill.sh", "vllm"],
                    check=False,
                )
                subprocess.run(
                    ["bash", "scripts/tools/slow_safe_pkill.sh", "VLLM"],
                    check=False,
                )
                subprocess.run(
                    ["bash", "scripts/tools/slow_safe_pkill.sh", "multiprocessing.spawn"],
                    check=False,
                )
            wall_time = time.time() - t0

            run_meta = {
                "model_key": model["key"],
                "model_name": model["name"],
                "model_path": model["path"],
                "tp": model["tp"],
                "dp": model["dp"],
                "gpus": model["gpus"],
                "data": args.data,
                "max_samples": args.max_samples,
                "max_tokens": args.max_tokens,
                "concurrency": args.concurrency,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "port": args.port,
                "infer_dir": str(infer_dir),
                "wall_time_sec": round(wall_time, 3),
                "timestamp": int(time.time()),
            }
            with run_meta_path.open("w", encoding="utf-8") as f:
                json.dump(run_meta, f, indent=2)
        else:
            with run_meta_path.open("r", encoding="utf-8") as f:
                run_meta = json.load(f)
            wall_time = run_meta.get("wall_time_sec", 0.0)

        expected_samples = 0
        total_glyphs = 0
        for sample in iter_jsonl(args.data):
            if args.max_samples > 0 and expected_samples >= args.max_samples:
                break
            total_glyphs += len(extract_text(sample))
            expected_samples += 1

        total_samples = 0
        error_samples = 0
        missing_usage = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        for obj in iter_jsonl(infer_dir):
            total_samples += 1
            if obj.get("error"):
                error_samples += 1
                continue
            usage = obj.get("usage") or {}
            if not usage:
                missing_usage += 1
                continue
            prompt_tokens += usage.get("prompt_tokens", 0) or 0
            completion_tokens += usage.get("completion_tokens", 0) or 0
            total_tokens += usage.get("total_tokens", 0) or 0

        sec_per_glyph = wall_time / total_glyphs if total_glyphs else 0.0
        glyphs_per_sec = total_glyphs / wall_time if wall_time else 0.0
        sec_per_token = (
            wall_time / completion_tokens if completion_tokens else 0.0
        )
        tokens_per_sec = (
            completion_tokens / wall_time if wall_time else 0.0
        )

        row = {
            "model_key": model["key"],
            "model_name": model["name"],
            "model_path": model["path"],
            "tp": model["tp"],
            "dp": model["dp"],
            "gpus": model["gpus"],
            "data": args.data,
            "max_samples": args.max_samples,
            "expected_samples": expected_samples,
            "processed_samples": total_samples,
            "error_samples": error_samples,
            "missing_usage": missing_usage,
            "total_glyphs": total_glyphs,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "wall_time_sec": round(wall_time, 3),
            "sec_per_glyph": round(sec_per_glyph, 6),
            "glyphs_per_sec": round(glyphs_per_sec, 3),
            "sec_per_token": round(sec_per_token, 6),
            "tokens_per_sec": round(tokens_per_sec, 3),
            "token_basis": "completion_tokens",
        }
        table.append(row)

    table_path = output_dir / "throughput_table.json"
    with table_path.open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)

    csv_path = output_dir / "throughput_table.csv"
    if table:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)

    print(f"Wrote {table_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
