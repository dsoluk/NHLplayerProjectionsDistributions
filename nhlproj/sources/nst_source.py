from typing import Optional, Dict, List
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from nhlproj.config.schema import NST_SCHEMA, NST_SYNONYMS, INFO_ONLY
from nhlproj.utils.columns import apply_synonyms, coerce_numeric, sanitize_ipp, derive_pos_group

VALID_SITS = ["5v5", "pp", "pk", "all"]


def build_nst_url(
    fromseason: str,
    thruseason: str,
    stype: int = 2,
    sit: str = "5v5",
    score: str = "all",
    stdoi: str = "std",
    rate: str = "y",
    team: str = "ALL",
    pos: str = "S",
    loc: str = "B",
    toi: int = 0,
    gpfilt: str = "none",
    fd: str = "",
    td: str = "",
    tgp: int = 410,
    lines: str = "single",
    draftteam: str = "ALL",
) -> str:
    base = "https://www.naturalstattrick.com/playerteams.php"
    sit = sit.lower()
    if sit not in VALID_SITS:
        raise ValueError(f"Invalid sit '{sit}'. Valid options: {VALID_SITS}")
    params = {
        "fromseason": fromseason,
        "thruseason": thruseason,
        "stype": stype,
        "sit": sit,
        "score": score,
        "stdoi": stdoi,
        "rate": rate,
        "team": team,
        "pos": pos,
        "loc": loc,
        "toi": toi,
        "gpfilt": gpfilt,
        "fd": fd,
        "td": td,
        "tgp": tgp,
        "lines": lines,
        "draftteam": draftteam,
    }
    return f"{base}?{urlencode(params)}"


def fetch_nst_playerteams(url: str) -> pd.DataFrame:
    """Fetch the NST playerteams table using pandas.read_html and return the main DataFrame."""
    try:
        tables = pd.read_html(url)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch NST data from URL: {e}")
    if not tables:
        raise RuntimeError("No tables found at NST URL")
    tbl = max(tables, key=lambda t: (t.shape[0] * t.shape[1]))
    return tbl


def normalize_nst_columns(df: pd.DataFrame, sit: str) -> pd.DataFrame:
    """
    Normalize/rename columns from NST for schema-driven analysis:
    - Apply synonym mapping so columns match schema names (e.g., "iSF/60" -> "Shots/60").
    - Coerce all schema columns (except InfoOnly) to numeric when present.
    - Create PPP from P for PP tables if missing.
    - Sanitize IPP infinities to 0 to avoid Excel issues.
    - Derive a coarse position group column `pos_group` from `Position`.
    """
    df_out = df.copy()

    # Apply synonyms so headers align with our schema
    df_out = apply_synonyms(df_out, NST_SYNONYMS)

    # PPP mapping for PP tables
    if sit.lower() == "pp" and "PPP" not in df_out.columns and "P" in df_out.columns:
        df_out["PPP"] = pd.to_numeric(df_out["P"], errors="coerce")

    # Coerce numerics for known schema columns that are not InfoOnly
    numeric_targets = [k for k, v in NST_SCHEMA.items() if v != INFO_ONLY and k in df_out.columns]
    df_out = coerce_numeric(df_out, numeric_targets)

    # Sanitize IPP infinities
    if "IPP" in df_out.columns:
        df_out = sanitize_ipp(df_out, column="IPP")

    # Derive position group
    df_out = derive_pos_group(df_out, position_col="Position", out_col="pos_group")

    return df_out
