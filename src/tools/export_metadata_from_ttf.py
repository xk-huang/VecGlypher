#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export font (TTF/OTF/TTC) metadata to JSON.

Examples
--------
# Single file -> stdout
python src/tools/export_metadata_from_ttf.py /path/MyFont.ttf --pretty

# Multiple files into one JSON
python src/tools/export_metadata_from_ttf.py fonts/*.otf -o meta.json

# Scan a directory (recursively) for fonts
python src/tools/export_metadata_from_ttf.py /path/to/fonts --recursive --pretty

# Scan with custom patterns (comma-separated)
python src/tools/export_metadata_from_ttf.py /fonts --recursive --pattern "*.ttf,*.otf,*.ttc" -o out.json
"""
from __future__ import annotations

import glob

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click

try:
    from fontTools.ttLib import TTCollection, TTFont
except ImportError as e:
    click.echo(
        "Missing dependency: fontTools. Install with `pip install fonttools`.", err=True
    )
    raise

MAC_EPOCH = datetime(1904, 1, 1)

NAME_ID_MAP = {
    0: "copyright",
    1: "family",
    2: "subfamily",
    3: "unique_id",
    4: "full_name",
    5: "version",
    6: "postscript_name",
    7: "trademark",
    8: "manufacturer",
    9: "designer",
    10: "designer_url",
    11: "description",
    12: "manufacturer_url",
    13: "license",
    14: "license_url",
    15: "reserved",
    16: "typographic_family",
    17: "typographic_subfamily",
    18: "compatible_full",
    19: "sample_text",
    20: "ps_cid_name",
    21: "wws_family",
    22: "wws_subfamily",
    25: "variations_ps_name_prefix",
}


def mac_timestamp_to_iso(ts: int | float | None) -> Optional[str]:
    if ts is None:
        return None
    try:
        return (MAC_EPOCH + timedelta(seconds=float(ts))).isoformat()
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def table_present(tt: TTFont, tag: str) -> bool:
    try:
        return tag in tt
    except Exception:
        return False


def extract_name_table(tt: TTFont) -> Dict[str, Any]:
    """Return both 'best' (debug) names and full raw name records."""
    out: Dict[str, Any] = {"best": {}, "records": []}
    if not table_present(tt, "name"):
        return out

    name_table = tt["name"]

    # Best/most useful fields (fontTools heuristics)
    for nid, key in NAME_ID_MAP.items():
        try:
            val = name_table.getDebugName(nid)
        except Exception:
            val = None
        if val is not None:
            out["best"][key] = str(val)

    # All raw name records
    try:
        for rec in name_table.names:
            try:
                s = rec.toUnicode()
            except Exception:
                # fall back to raw bytes decode
                try:
                    s = rec.string.decode("utf-8", errors="replace")
                except Exception:
                    s = None
            out["records"].append(
                {
                    "name_id": rec.nameID,
                    "name_key": NAME_ID_MAP.get(rec.nameID, f"name_id_{rec.nameID}"),
                    "platform_id": rec.platformID,
                    "plat_enc_id": getattr(rec, "platEncID", None),
                    "lang_id": getattr(rec, "langID", None),
                    "string": s,
                }
            )
    except Exception:
        # ignore if 'names' is oddly formatted
        pass

    return out


def extract_head(tt: TTFont) -> Dict[str, Any]:
    if not table_present(tt, "head"):
        return {}
    t = tt["head"]
    return {
        "units_per_em": getattr(t, "unitsPerEm", None),
        "font_revision": safe_float(getattr(t, "fontRevision", None)),
        "created": mac_timestamp_to_iso(getattr(t, "created", None)),
        "modified": mac_timestamp_to_iso(getattr(t, "modified", None)),
        "flags": getattr(t, "flags", None),
        "xMin": getattr(t, "xMin", None),
        "yMin": getattr(t, "yMin", None),
        "xMax": getattr(t, "xMax", None),
        "yMax": getattr(t, "yMax", None),
        "mac_style": (
            getattr(tt["OS/2"], "fsSelection", None)
            if table_present(tt, "OS/2")
            else None
        ),
        "check_sum_adjustment": getattr(t, "checkSumAdjustment", None),
        "index_to_loc_format": getattr(t, "indexToLocFormat", None),
    }


def extract_hhea(tt: TTFont) -> Dict[str, Any]:
    if not table_present(tt, "hhea"):
        return {}
    t = tt["hhea"]
    return {
        "ascent": getattr(t, "ascent", None),
        "descent": getattr(t, "descent", None),
        "line_gap": getattr(t, "lineGap", None),
        "advance_width_max": getattr(t, "advanceWidthMax", None),
        "min_left_side_bearing": getattr(t, "minLeftSideBearing", None),
        "min_right_side_bearing": getattr(t, "minRightSideBearing", None),
        "x_max_extent": getattr(t, "xMaxExtent", None),
        "caret_slope_rise": getattr(t, "caretSlopeRise", None),
        "caret_slope_run": getattr(t, "caretSlopeRun", None),
        "number_of_hmetrics": getattr(t, "numberOfHMetrics", None),
    }


def extract_post(tt: TTFont) -> Dict[str, Any]:
    if not table_present(tt, "post"):
        return {}
    t = tt["post"]
    return {
        "italic_angle": safe_float(getattr(t, "italicAngle", None)),
        "underline_position": getattr(t, "underlinePosition", None),
        "underline_thickness": getattr(t, "underlineThickness", None),
        "is_fixed_pitch": bool(getattr(t, "isFixedPitch", 0)),
    }


def extract_os2(tt: TTFont) -> Dict[str, Any]:
    if not table_present(tt, "OS/2"):
        return {}
    t = tt["OS/2"]
    fs_sel = getattr(t, "fsSelection", None)

    def bit(b: int) -> Optional[bool]:
        if fs_sel is None:
            return None
        return bool(fs_sel & (1 << b))

    out = {
        "version": getattr(t, "version", None),
        "weight_class": getattr(t, "usWeightClass", None),
        "width_class": getattr(t, "usWidthClass", None),
        "vendor_id": getattr(t, "achVendID", None),
        "fs_type": getattr(t, "fsType", None),  # embedding permissions flags
        "fs_selection": fs_sel,
        "fs_selection_flags": {
            "ITALIC": bit(0),
            "UNDERSCORE": bit(1),
            "NEGATIVE": bit(2),
            "OUTLINED": bit(3),
            "STRIKEOUT": bit(4),
            "BOLD": bit(5),
            "REGULAR": bit(6),
            "USE_TYPO_METRICS": bit(7),
            "WWS": bit(8),
            "OBLIQUE": bit(9),
        },
        "typo_ascender": getattr(t, "sTypoAscender", None),
        "typo_descender": getattr(t, "sTypoDescender", None),
        "typo_line_gap": getattr(t, "sTypoLineGap", None),
        "win_ascent": getattr(t, "usWinAscent", None),
        "win_descent": getattr(t, "usWinDescent", None),
        "cap_height": getattr(
            t,
            "sCapHeight",
            None,
        ),
        "x_height": getattr(t, "sxHeight", None),
        # "panose": list(getattr(t, "panose", [])) if hasattr(t, "panose") else None,
        "unicode_range": [
            getattr(t, "ulUnicodeRange1", None),
            getattr(t, "ulUnicodeRange2", None),
            getattr(t, "ulUnicodeRange3", None),
            getattr(t, "ulUnicodeRange4", None),
        ],
        "codepage_range": [
            getattr(t, "ulCodePageRange1", None),
            getattr(t, "ulCodePageRange2", None),
        ],
    }
    return out


def extract_cmap(tt: TTFont) -> Dict[str, Any]:
    if not table_present(tt, "cmap"):
        return {}
    cmap = tt["cmap"]
    encodings = []
    codepoints: set[int] = set()
    try:
        for sub in cmap.tables:
            try:
                cps = set(sub.cmap.keys())
            except Exception:
                cps = set()
            encodings.append(
                {
                    "platform_id": getattr(sub, "platformID", None),
                    "plat_enc_id": getattr(sub, "platEncID", None),
                    "format": getattr(sub, "format", None),
                    "length": len(cps),
                }
            )
            codepoints |= cps
    except Exception:
        pass
    return {"total_unique_codepoints": len(codepoints), "encodings": encodings}


def extract_fvar(tt: TTFont) -> Dict[str, Any]:
    if not table_present(tt, "fvar"):
        return {}
    fvar = tt["fvar"]
    axes = []
    for ax in fvar.axes:
        axes.append(
            {
                "tag": ax.axisTag,
                "min": safe_float(ax.minValue),
                "default": safe_float(ax.defaultValue),
                "max": safe_float(ax.maxValue),
                "name": (
                    tt["name"].getDebugName(ax.axisNameID)
                    if table_present(tt, "name")
                    else None
                ),
            }
        )
    instances = []
    for inst in getattr(fvar, "instances", []):
        instances.append(
            {
                "name": (
                    tt["name"].getDebugName(inst.subfamilyNameID)
                    if table_present(tt, "name")
                    else None
                ),
                "coordinates": {k: safe_float(v) for k, v in inst.coordinates.items()},
                "ps_name_id": getattr(inst, "postscriptNameID", None),
            }
        )
    return {"axes": axes, "instances": instances}


def extract_stat(tt: TTFont) -> Dict[str, Any]:
    if not table_present(tt, "STAT"):
        return {}
    stat = tt["STAT"].table
    out: Dict[str, Any] = {}
    try:
        axes = []
        for ax in getattr(stat.DesignAxisRecord, "Axis", []):
            axes.append(
                {
                    "tag": ax.AxisTag,
                    "ordering": ax.AxisOrdering,
                    "name_id": ax.AxisNameID,
                    "name": (
                        tt["name"].getDebugName(ax.AxisNameID)
                        if table_present(tt, "name")
                        else None
                    ),
                }
            )
        out["design_axes"] = axes
    except Exception:
        pass
    try:
        values = []
        for val in getattr(stat.AxisValueArray, "AxisValue", []):
            values.append(
                {
                    "format": getattr(val, "Format", None),
                    "axis_index": getattr(val, "AxisIndex", None),
                    "value_name_id": getattr(val, "ValueNameID", None),
                    "value": getattr(val, "Value", None),
                }
            )
        out["axis_values"] = values
    except Exception:
        pass
    return out


def extract_misc(tt: TTFont) -> Dict[str, Any]:
    present = set(tt.keys())
    return {
        "tables_present": sorted(present),
        "num_glyphs": (
            getattr(tt["maxp"], "numGlyphs", None)
            if table_present(tt, "maxp")
            else None
        ),
        "outline_format": (
            "glyf"
            if "glyf" in present
            else (
                "cff2" if "CFF2" in present else ("cff" if "CFF " in present else None)
            )
        ),
        "has_svg": "SVG " in present,
        "has_color": bool(
            {"COLR", "CPAL"} & present
            or {"CBDT", "CBLC"} & present
            or {"sbix"} & present
        ),
    }


def extract_font_metadata(tt: TTFont) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    meta["name"] = extract_name_table(tt)
    meta["head"] = extract_head(tt)
    meta["hhea"] = extract_hhea(tt)
    meta["post"] = extract_post(tt)
    meta["os2"] = extract_os2(tt)
    meta["cmap"] = extract_cmap(tt)
    meta["fvar"] = extract_fvar(tt)
    meta["stat"] = extract_stat(tt)
    meta["misc"] = extract_misc(tt)
    return meta


def iter_font_paths(
    inputs: Iterable[str], recursive: bool, patterns: List[str]
) -> List[str]:
    files: List[str] = []
    pats = [p.strip() for p in patterns if p.strip()]
    for inp in inputs:
        if os.path.isdir(inp):
            if recursive:
                for root, dirs, _ in os.walk(inp):
                    for p in pats:
                        files.extend(glob.glob(os.path.join(root, p)))
            else:
                for p in pats:
                    files.extend(glob.glob(os.path.join(inp, p)))
        else:
            files.append(inp)
    # Dedup while preserving order
    seen = set()
    out = []
    for f in files:
        f = os.path.abspath(f)
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def extract_from_file(path: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Return list of (label, metadata) entries. Handles TTC/OTC collections."""
    ext = os.path.splitext(path)[1].lower()
    entries: List[Tuple[str, Dict[str, Any]]] = []
    try:
        if ext in (".ttc", ".otc"):
            coll = TTCollection(path, lazy=False)
            for idx, tt in enumerate(coll.fonts):
                label = f"{path}#{idx}"
                try:
                    meta = extract_font_metadata(tt)
                    entries.append((label, meta))
                finally:
                    tt.close()
            coll.close()
        else:
            tt = TTFont(path, lazy=False)
            try:
                meta = extract_font_metadata(tt)
                entries.append((path, meta))
            finally:
                tt.close()
    except Exception as e:
        entries.append((path, {"error": str(e)}))
    return entries


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument(
    "inputs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=str),
)
@click.option(
    "-o",
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, writable=True, path_type=str),
    help="Output JSON file (default: print to stdout).",
)
@click.option(
    "--recursive",
    is_flag=True,
    help="Recurse into directories when inputs include directories.",
)
@click.option(
    "--pattern",
    default="*.ttf,*.otf,*.ttc,*.otc",
    show_default=True,
    help="Comma-separated glob patterns to match fonts when scanning directories.",
)
@click.option("--pretty", is_flag=True, help="Pretty-print JSON (indent=2).")
def cli(
    inputs: Tuple[str, ...],
    out_path: Optional[str],
    recursive: bool,
    pattern: str,
    pretty: bool,
) -> None:
    """
    Export metadata of fonts (TTF/OTF/TTC) to JSON.

    INPUTS can be font files and/or directories. If a TTC/OTC is provided,
    all faces are exported with labels like 'path/to/font.ttc#0', '#1', ...
    """
    patterns = [p.strip() for p in pattern.split(",") if p.strip()]
    font_paths = iter_font_paths(inputs, recursive=recursive, patterns=patterns)
    if not font_paths:
        click.echo(
            "No font files found matching the provided inputs/patterns.", err=True
        )
        sys.exit(2)

    results: List[Dict[str, Any]] = []
    for p in font_paths:
        for label, meta in extract_from_file(p):
            results.append(
                {
                    "source": label,
                    "metadata": meta,
                }
            )

    payload = json.dumps(results, ensure_ascii=False, indent=2 if pretty else None)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(payload)
        click.echo(f"Wrote metadata for {len(results)} font face(s) -> {out_path}")
    else:
        click.echo(payload)


if __name__ == "__main__":
    cli()
