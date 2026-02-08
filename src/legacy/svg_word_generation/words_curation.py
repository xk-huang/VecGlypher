"""
python -m src.svg_word_generation.words_curation
"""

from pathlib import Path

from wordfreq import top_n_list


def main():
    output_dir = Path("data/processed/words_curation/")
    output_dir.mkdir(exist_ok=True, parents=True)

    # minimal character sets
    words = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    output_path = output_dir / "minimal_char.txt"
    with open(output_path, "w") as f:
        for word in words:
            f.write(word + "\n")
    print(f"Minimal Characters saved to {output_path}")

    # pangrams
    words = ["The quick brown fox jumps over the lazy dog"]

    output_path = output_dir / "pangrams.txt"
    with open(output_path, "w") as f:
        for word in words:
            f.write(word + "\n")
    print(f"Pangrams saved to {output_path}")

    # top n words
    lang = "en"
    n = 2000
    wordlist = "large"
    ascii_only = False
    words = top_n_list(
        lang=lang,
        n=n,
        wordlist=wordlist,
        ascii_only=ascii_only,
    )

    output_path = output_dir / f"words-{lang}-{n}.txt"
    with open(output_path, "w") as f:
        for word in words:
            f.write(word + "\n")
    print(f"Words saved to {output_path}")


if __name__ == "__main__":
    main()
