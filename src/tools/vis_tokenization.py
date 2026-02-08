"""
HF_HUB_DISABLE_XET=1 with-proxy python src/tools/vis_tokenization.py
"""

from ast import literal_eval

import click

from transformers import AutoTokenizer


def vis_tokenization(tokenizer, text, decode=False):
    if decode:
        breakpoint()
        return tokenizer.decode(
            text, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    # Tokenize the text
    encoded = tokenizer(text, return_tensors="pt")
    # Get token IDs and convert to list
    token_ids = encoded["input_ids"][0].tolist()
    # Decode each token individually
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
    return tokens


@click.command()
@click.option("--model_name", default="qwen/Qwen3-8B")
@click.argument("text", default='"Hello, world!"')
@click.option("--pdb", is_flag=True, default=False)
@click.option("--decode", is_flag=True, default=False)
def main(model_name, text, pdb, decode):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = literal_eval(text)
    tokens = vis_tokenization(tokenizer, text, decode)
    print(f"Tokens: {tokens}")

    if pdb:
        # fmt: off
        from IPython import embed; embed()
        # fmt: on

        sep_char_token = "<|sep_char|>"
        additional_special_tokens = [sep_char_token]
        tokenizer.add_special_tokens(
            {"additional_special_tokens": additional_special_tokens},
            replace_additional_special_tokens=False,
        )
        print(f"tokenizer.special_tokens_map: {tokenizer.special_tokens_map}")

        text = "storage_cli"
        print(f"tokens: {vis_tokenization(tokenizer, text)}")
        sep_char_text = sep_char_token.join(list(text))
        print(f"tokens: {vis_tokenization(tokenizer, sep_char_text)}")


if __name__ == "__main__":
    main()
