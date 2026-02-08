"""
python -m src.svg_glyph_gen_v2.filter_invalid_fonts
"""

import json
import os
import string
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import click
from fontTools.ttLib import TTFont
from tqdm import tqdm

from .utils import load_jsonl, prepare_output_dir_and_logger, write_jsonl

# Define alphanumeric characters
VALID_CHAR_SET = string.ascii_letters + string.digits

# [NOTE](xk) some fonts does not have full string.punctuation
# In oxford 3000/5000 list, there are words with space " ", hyphen "-", or "'"
# we do not use string.punctuation here as some of them like `"`"` are rare to use

VALID_CHAR_SET += "-'"
# VALID_CHAR_SET += string.punctuation + " "


@click.command()
@click.option(
    "--input_metadata_jsonl",
    type=click.Path(exists=True),
    # default="data/processed/google_font_metadata/",
    required=True,
)
@click.option(
    "--input_google_font_dir",
    type=click.Path(),
    # default="data/google_fonts/ofl",
    required=True,
)
@click.option(
    "--output_dir",
    type=click.Path(),
    # default="data/processed/google_font_metadata-filter_invalid",
    required=True,
)
@click.option("--max_num_fonts", type=int, default=None)
@click.option("--num_workers", type=int, default=20)
@click.option("--overwrite", is_flag=True, default=False)
def main(**kargs):
    args = SimpleNamespace(**kargs)

    font_filter = FontFilter(args)
    font_filter.filter_invalid_fonts()


class FontFilter:
    def __init__(self, args):
        self.args = args

        # prepare output dir and logger
        should_skip, logger = prepare_output_dir_and_logger(
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        if should_skip:
            exit()
        self.output_dir = Path(args.output_dir)
        self.logger = logger

        self.font_sub_dir = Path(self.args.input_google_font_dir)
        self.num_workers = args.num_workers

        # Setup logger

    def filter_invalid_fonts(self):
        metadata = []
        input_metadata_jsonl = self.args.input_metadata_jsonl
        input_metadata_jsonl = Path(input_metadata_jsonl)
        if not input_metadata_jsonl.exists():
            raise FileNotFoundError(f"{input_metadata_jsonl} does not exist.")

        metadata = load_jsonl(input_metadata_jsonl)

        if self.args.max_num_fonts:
            metadata = metadata[: self.args.max_num_fonts]
            self.logger.info(f"Limit to {self.args.max_num_fonts} fonts.")

        failed_font_idx_list = self.get_invalid_font_idx_list(metadata)

        # save filtered metadata
        filtered_metadata = [
            metadata_item
            for idx, metadata_item in enumerate(metadata)
            if idx not in failed_font_idx_list
        ]
        num_fonts_before = len(metadata)
        num_fonts_after = len(filtered_metadata)
        self.logger.info(f"Number of fonts: {num_fonts_before} -> {num_fonts_after}")

        filtered_metadat_jsonl = (
            self.output_dir / "google_font_metadata.filter_invalid.jsonl"
        )
        write_jsonl(filtered_metadata, filtered_metadat_jsonl, logger=self.logger)
        self.logger.info(f"Write to: {filtered_metadat_jsonl}")

    def _is_font_valid(self, idx, font_file_path):
        return idx, is_font_valid(font_file_path, self.logger)

    def get_invalid_font_idx_list(self, metadata):
        results = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for idx, metadata_item in enumerate(metadata):
                font_file_name = metadata_item["filename"]
                font_family_dir_name = metadata_item["font_family_dir_name"]
                font_path = self.font_sub_dir / font_family_dir_name / font_file_name
                futures.append(executor.submit(self._is_font_valid, idx, font_path))

            pbar = tqdm(total=len(futures))
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)

        failed_font_idx_list = [idx for idx, is_success in results if not is_success]
        failed_fonts = []
        for idx in failed_font_idx_list:
            metadata_item = metadata[idx]
            failed_fonts.append(Path(metadata_item["filename"]).name)

        # save failed fonts with details
        def _write_fonts_list(font_list, output_file_name):
            font_list = sorted(font_list)

            output_sub_dir = "failed_fonts"
            output_dir = self.output_dir / output_sub_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            failed_font_name_path = output_dir / output_file_name

            with open(failed_font_name_path, "w") as f:
                for font_file_name in font_list:
                    f.write(font_file_name + "\n")
            self.logger.info(
                f"fonts: {len(font_list)}. Write to {failed_font_name_path}."
            )

        _write_fonts_list(set(failed_fonts), "failed_fonts.txt")
        _write_fonts_list(set(MANUAL_FAILED_FONTS_SET), "manual_failed_fonts.txt")
        _write_fonts_list(
            set(failed_fonts) - set(MANUAL_FAILED_FONTS_SET),
            "diff_manual_failed_fonts.txt",
        )
        _write_fonts_list(
            set(MANUAL_FAILED_FONTS_SET) - set(failed_fonts),
            "diff_auto_failed_fonts.txt",
        )

        # add manual failed fonts
        self.logger.info(f"Auto-detected failed fonts: {len(failed_fonts)}")
        failed_fonts += MANUAL_FAILED_FONTS_SET
        failed_fonts = set(failed_fonts)
        self.logger.info(
            f"Auto-detected + manually selected failed fonts : {len(failed_fonts)}"
        )

        # NOTE: we use index instead of font name. Add those manually failed font index
        # build MANUAL_FAILED_FONTS_SET to metadata index
        manual_failed_font_idx_list = []
        for idx, metadata_item in enumerate(metadata):
            font_file_name = metadata_item["filename"]
            if font_file_name in MANUAL_FAILED_FONTS_SET:
                manual_failed_font_idx_list.append(idx)
        failed_font_idx_list = set(failed_font_idx_list + manual_failed_font_idx_list)

        return failed_font_idx_list


def is_font_valid(font_path, logger=None):
    if is_special_purpose_font(font_path):
        return False

    return has_alphanumeric_glyphs(font_path, logger=logger)


def is_special_purpose_font(font_path, verbose=False):
    """
    Check if font is likely a special-purpose font (barcode, symbols, etc.)
    """
    filename = Path(font_path).name.lower()

    special_indicators = [
        # Barcode fonts
        "barcode",
        # Symbol fonts
        "symbol",
        "emoji",
        # Redaction fonts
        "redacted",
        "block",
        # Musical notation
        "musical",
        "music",
        # Mathematical symbols
        "math",
        # Decorative fonts
        "decorative",  # Only captical character
        # Blank fonts
        "blank",
        # other indicators
        "flowcircular",
        "redacted",
        "flowrounded",
        "linefont",
        "wavefont",
        "yarndings",
    ]
    if verbose:
        for indicator in special_indicators:
            if indicator in filename:
                print(f"Found {indicator} in {filename}")
                return True
        return False

    return any(indicator in filename for indicator in special_indicators)


def has_alphanumeric_glyphs(font_path, verbose=False, logger=None):
    """
    Check if a font contains all alphanumeric characters (a-zA-Z0-9) and `-' `.

    Args:
        font_path (str): Path to the TTF font file

    Returns:
        bool: True if font contains all alphanumeric glyphs, False otherwise
    """
    try:
        font = TTFont(font_path)
        cmap = font.getBestCmap()

        # Check if all alphanumeric characters are present
        for char in VALID_CHAR_SET:
            if ord(char) not in cmap:
                if verbose:
                    print(f"Missing glyph for '{char}' in {font_path}")
                return False

        return True

    except Exception as e:
        if logger:
            logger.error(f"Error processing {font_path}: {e}")
        return False


# [NOTE](xk) manually collected failed fonts by skimming through the rendered pangram SVGs
MANUAL_FAILED_FONTS_SET = {
    # Tofo
    "NotoSansDevanagariUI-ExtraLight.ttf",
    "NotoSansBengaliUI[wdth,wght].ttf",
    "Content-Regular.ttf",
    "NotoSansGujaratiUI-SemiBold.ttf",
    "NotoSansArabicUI[wdth,wght].ttf",
    "NotoSerifMyanmar-Black.ttf",
    "NotoSansGujaratiUI-Bold.ttf",
    "NotoSerifMyanmar-ExtraLight.ttf",
    "NotoSansDevanagariUI-Thin.ttf",
    "NotoSansDevanagariUI-Light.ttf",
    "NotoSansKhmerUI[wdth,wght].ttf",
    "NotoSerifMyanmar-Light.ttf",
    "NotoEmoji[wght].ttf",
    "NotoSansKannadaUI[wdth,wght].ttf",
    "NotoSerifMyanmar-Medium.ttf",
    "NotoSansDevanagariUI-SemiBold.ttf",
    "NotoSansDevanagariUI-Regular.ttf",
    "NotoSerifMyanmar-Bold.ttf",
    "NotoSansDevanagariUI-Bold.ttf",
    "NotoSansGujaratiUI-Black.ttf",
    "NotoSerifMyanmar-SemiBold.ttf",
    "Ponnala-Regular.ttf",
    "NotoSansGurmukhiUI[wdth,wght].ttf",
    "NotoSerifMyanmar-Regular.ttf",
    "NotoSansGujaratiUI-Regular.ttf",
    "NotoSansGujaratiUI-ExtraLight.ttf",
    "NotoSansGujaratiUI-Thin.ttf",
    "Content-Bold.ttf",
    "NotoSansDevanagariUI-ExtraBold.ttf",
    "NotoSerifMyanmar-Thin.ttf",
    "Chenla.ttf",
    "NotoSerifMyanmar-ExtraBold.ttf",
    "NotoSansGujaratiUI-ExtraBold.ttf",
    "NotoSansDevanagariUI-Medium.ttf",
    "NotoSerifNyiakengPuachueHmong[wght].ttf",
    "NotoSansGujaratiUI-Medium.ttf",
    "NotoSansDevanagariUI-Black.ttf",
    "Siemreap.ttf",
    "NotoSansGujaratiUI-Light.ttf",
    # Dev
    "RedactedScript-Bold.ttf",
    "Redacted-Regular.ttf",
    "FlowRounded-Regular.ttf",
    "Linefont[wdth,wght].ttf",
    "RedactedScript-Regular.ttf",
    "RedactedScript-Light.ttf",
    "Wavefont[ROND,YELA,wght].ttf",
    "FlowBlock-Regular.ttf",
    "LibreBarcode39ExtendedText-Regular.ttf",
    "FlowCircular-Regular.ttf",
    # Dev empty
    "AdobeBlank-Regular.ttf",
    "Yarndings20-Regular.ttf",
    # Symbol
    "Yarndings12-Regular.ttf",
    "Yarndings12Charted-Regular.ttf",
    "NotoColorEmoji-Regular.ttf",
    "Yarndings20Charted-Regular.ttf",
    # Bar Code
    "LibreBarcode39Text-Regular.ttf",
    "LibreBarcode39Extended-Regular.ttf",
    "LibreBarcodeEAN13Text-Regular.ttf",
    "LibreBarcode128-Regular.ttf",
    "LibreBarcode39-Regular.ttf",
    "LibreBarcode128Text-Regular.ttf",
    # Empty
    "KarlaTamilInclined-Regular.ttf",
    "Khmer.ttf",
    "KarlaTamilUpright-Regular.ttf",
    "KarlaTamilUpright-Bold.ttf",
    "Phetsarath-Bold.ttf",
    "Phetsarath-Regular.ttf",
    "KarlaTamilInclined-Bold.ttf",
    # Missing characters
    "NotoTraditionalNushu[wght].ttf",  # missing hyphen `-`
}


if __name__ == "__main__":
    main()
