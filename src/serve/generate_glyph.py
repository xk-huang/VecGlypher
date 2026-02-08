"""
Download model first:

python scripts/tools/download_model_from_storage.py -i workspace/hf_downloads/Qwen/Qwen3-1.7B -o saves/Qwen3-1.7B

model_path="saves/Qwen3-1.7B"
python -m sglang.launch_server \
    --model-path "${model_path}" \
    --nccl-port 10000 \
    --port 30000 \
    --tp 1 \
    --dp 1

model_path="saves/Qwen3-1.7B"
python -m src.serve.generate_glyph "Text content: a" --model "${model_path}"
python -m src.serve.generate_glyph "Text content: a<|SEP|>b<|SEP|>c" --model "${model_path}"


The ports on OnDemand are randomly opened and varied during the lifetime of the server.
Use lower ports which are more likely to be opened.
See:
- https://github.com/sgl-project/sglang/pull/10620
- https://github.com/sgl-project/sglang/pull/10619
"""

import json

import sys

import click
from openai import OpenAI

from ..svg_glyph_gen_v2.build_sft_data_v2 import SYSTEM_PROMPT
from ..svg_glyph_gen_v2.svg_simplifier import SVGSimplifier
from ..svg_glyph_gen_v2.utils import timer_display


def _client(server: str, api_key: str):
    # vLLM ignores the key; but the OpenAI client requires something
    return OpenAI(base_url=server.rstrip("/") + "/v1", api_key=api_key or "EMPTY")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("prompt", required=False)
@click.option(
    "--server",
    default="http://localhost:30000",
    show_default=True,
    help="Base URL of the vLLM server (without /v1).",
)
@click.option(
    "--model",
    required=True,
    help="Model name as exposed by vLLM (--served-model-name).",
)
@click.option("--system", default=SYSTEM_PROMPT, help="Optional system message.")
@click.option("--temperature", default=0.2, show_default=True, type=float)
@click.option("--top-p", default=1.0, show_default=True, type=float)
@click.option("--max-tokens", default=8192, show_default=True, type=int)
@click.option("--stream/--no-stream", default=False, show_default=True)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    default="",
    help="Unused by vLLM; provided for compatibility.",
)
@click.option(
    "--decode/--no-decode", default=True, show_default=True, help="Decode path to SVG."
)
@click.option("--json-out", is_flag=True, help="Print raw JSON response.")
def chat(
    prompt,
    server,
    model,
    system,
    temperature,
    top_p,
    max_tokens,
    stream,
    api_key,
    decode,
    json_out,
):
    """
    Simple CLI to chat with a vLLM OpenAI-compatible server.

    PROMPT can be passed as an argument or piped via stdin (use '-' or omit PROMPT).
    """
    # Read prompt from stdin if needed
    if not prompt or prompt == "-":
        click.echo("Enter prompt:", nl=False)
        prompt = sys.stdin.read().strip()
        if not prompt:
            click.echo("No prompt provided.", err=True)
            sys.exit(1)
    click.echo(f"Prompt: {prompt}")
    client = _client(server, api_key)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if stream:
        # Stream tokens to stdout
        with client.chat.completions.stream(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ) as stream_resp:
            for event in stream_resp:
                if event.type == "chunk":
                    # update `.data` to `.chunk`
                    delta = event.chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        click.echo(content, nl=False)
                elif event.type == "error":
                    click.echo(f"\n[error] {event.data}", err=True)
            click.echo()  # newline at end
    else:
        with timer_display():
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        if json_out:
            click.echo("========================")
            click.echo(json.dumps(resp.model_dump(), ensure_ascii=False, indent=2))
            click.echo("========================")
        else:
            click.echo("========================")
            click.echo(resp.choices[0].message.content)
            click.echo("========================")

        if decode:
            svg_simplifier = SVGSimplifier()
            svg_str = svg_simplifier.decode(resp.choices[0].message.content)
            click.echo("========================")
            click.echo(svg_str)
            click.echo("========================")


if __name__ == "__main__":
    chat()
