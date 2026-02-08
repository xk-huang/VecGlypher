"""
sft_base_dir=data/processed/filtered_sft/250903-alphanumeric/
split_names=(
  "ood_font_family_decon"
)
for split_name in ${split_names[@]}; do
TOKENIZERS_PARALLELISM=true python -m src.tools.stat_token_len_for_paper \
--dataset_dir ${sft_base_dir} \
--dataset_name ${split_name} \
--output_dir misc/stat_token_len_for_paper/google_fonts-filtered_sft/${split_name}
done

GPT-5 https://openrouter.ai/openai/gpt-5
python scripts/tools/compute_token_price.py -i 1.25 -o 10

GPT-5 Mini https://openrouter.ai/openai/gpt-5-mini
python scripts/tools/compute_token_price.py -i 0.25 -o 2

GPT-5 Nanohttps://openrouter.ai/openai/gpt-5-nano
python scripts/tools/compute_token_price.py -i 0.05 -o 0.4

Claude Haiku 4.5 https://openrouter.ai/anthropic/claude-haiku-4.5
python scripts/tools/compute_token_price.py -i 1 -o 5

Gemini 2.5 Flash https://openrouter.ai/google/gemini-2.5-flash
python scripts/tools/compute_token_price.py -i 0.3 -o 2.5

Gemini 2.5 Pro https://openrouter.ai/google/gemini-2.5-pro
python scripts/tools/compute_token_price.py -i 1.25 -o 10

https://openrouter.ai/anthropic/claude-sonnet-4.5
python scripts/tools/compute_token_price.py -i 3 -o 15
"""

import json
import pprint
from collections import OrderedDict
from types import SimpleNamespace

import click


@click.command()
@click.option("--input_token_price", "-i", type=float)
@click.option("--output_token_price", "-o", type=float)
@click.option("--price_unit", "-u", type=int, default=1_000_000)
@click.option(
    "--input_stat_dict_json",
    type=str,
    default="misc/stat_token_len_for_paper/google_fonts-filtered_sft/ood_font_family_decon/stat_dict.json",
)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    input_token_price = args.input_token_price
    output_token_price = args.output_token_price
    price_unit = args.price_unit
    input_stat_dict_json = args.input_stat_dict_json

    with open(input_stat_dict_json, "r") as f:
        data = json.load(f)
    total_input_token_len = data["total_input_token_len"]
    total_output_token_len = data["total_output_token_len"]
    avg_input_token_len = data["avg_input_token_len"]
    avg_output_token_len = data["avg_output_token_len"]
    num_samples = data["num_samples"]

    total_input_cost = total_input_token_len / price_unit * input_token_price
    total_output_cost = total_output_token_len / price_unit * output_token_price
    avg_input_cost = total_input_cost / num_samples
    avg_output_cost = total_output_cost / num_samples
    cost_dict = {
        "total_input_cost": total_input_cost,
        "total_output_cost": total_output_cost,
        "avg_input_cost": avg_input_cost,
        "avg_output_cost": avg_output_cost,
        "num_samples": num_samples,
        "total_input_cost @ 1K": avg_input_cost * 1000,
        "total_output_cost @ 1K": avg_output_cost * 1000,
    }
    cost_dict = OrderedDict(cost_dict)
    pprint.pprint(cost_dict)


if __name__ == "__main__":
    main()
