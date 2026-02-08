"""
python google_font_processor/filter_failed_fonts.py
"""

import json

FAILED_FONTS_SET = {
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
}


def main():
    input_metadata_jsonl = "data/google_font_processor/google_font_metadata.jsonl"
    metadata_list = []
    with open(input_metadata_jsonl, "r") as f:
        for line in f:
            metadata_list.append(json.loads(line))

    filtered_metadata_list = []
    for metadata in metadata_list:
        if metadata["filename"] not in FAILED_FONTS_SET:
            filtered_metadata_list.append(metadata)
    num_failed_fonts = len(metadata_list) - len(filtered_metadata_list)
    print(
        f"Keep {len(filtered_metadata_list)} out of {len(metadata_list)}; Filtered number of failed fonts: {num_failed_fonts}"
    )

    output_metadata_jsonl = (
        "data/google_font_processor/google_font_metadata.filtered.jsonl"
    )
    with open(output_metadata_jsonl, "w", buffering=8192 * 16) as f:
        for metadata in filtered_metadata_list:
            f.write(json.dumps(metadata) + "\n")


if __name__ == "__main__":
    main()
