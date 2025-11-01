from typing import Dict, List
import numpy as np
import pandas as pd


def apply_synonyms(df: pd.DataFrame, synonyms: Dict[str, str]) -> pd.DataFrame:
    """Rename columns in-place using the provided synonyms map (incoming -> canonical).
    If multiple incoming names map to the same canonical and both are present, the first
    encountered is kept. Returns the DataFrame for chaining.
    """
    rename_map: Dict[str, str] = {}
    seen_targets: set = set()
    for incoming, target in synonyms.items():
        if incoming in df.columns and target not in seen_targets:
            rename_map[incoming] = target
            seen_targets.add(target)
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    return df


def coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Coerce listed columns to numeric if they exist.
    Keeps strings as NaN when conversion fails.
    """
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def coerce_specific_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Coerce specific columns to numeric, stripping commas/spaces if needed."""
    converted = []
    for col in columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                ser = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
                df[col] = pd.to_numeric(ser, errors='coerce')
                converted.append(col)
    if converted:
        print(f"Coerced to numeric: {converted}")
    return df


essentially_inf = [np.inf, -np.inf]

def sanitize_ipp(df: pd.DataFrame, column: str = "IPP") -> pd.DataFrame:
    """Replace +/-inf values in the IPP column with 0, leaving NaNs intact."""
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].replace(essentially_inf, 0)
    return df


def derive_pos_group(df: pd.DataFrame, position_col: str = "Position", out_col: str = "pos_group") -> pd.DataFrame:
    """Derive a coarse position group per row.
    Rules:
      - Any row that includes a defense designation (contains 'D') and also a forward (C/L/R) -> 'B' (both).
      - Only defense (contains 'D' and no C/L/R) -> 'D'.
      - Any forward only (contains any of C/L/R, but not D) -> 'F'.
      - Otherwise -> NaN.
    Position values may be like 'C', 'LW', 'RW', 'D', 'LD', 'RD', or comma/space separated lists 'C,LW'.
    """
    if position_col not in df.columns:
        df[out_col] = np.nan
        return df

    def parse_tokens(val: str) -> List[str]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        s = str(val).upper().replace('/', ',').replace('|', ',')
        # Normalize common abbreviations
        s = s.replace('LD', 'D').replace('RD', 'D').replace('LW', 'L').replace('RW', 'R')
        tokens = [t.strip() for t in s.replace(' ', ',').split(',') if t.strip()]
        # Expand combined like "C,RW" already split by commas above
        return tokens

    f_set = {'C', 'L', 'R'}

    groups: List[str] = []
    for v in df[position_col].values:
        toks = set(parse_tokens(v))
        has_d = 'D' in toks
        has_f = bool(f_set & toks)
        if has_d and has_f:
            groups.append('B')
        elif has_d:
            groups.append('D')
        elif has_f:
            groups.append('F')
        else:
            groups.append(np.nan)
    df[out_col] = groups
    return df
