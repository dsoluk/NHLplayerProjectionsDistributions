import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from nhlproj.analysis.fitting import FitResult


def _get_scipy_dist(name: str):
    if name == "Normal":
        return stats.norm
    if name == "LogNormal":
        return stats.lognorm
    if name == "Gamma":
        return stats.gamma
    raise ValueError(f"Unsupported distribution name: {name}")


def value_score_from_fit(x: float, fit: FitResult, method: str = "cdf_z") -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None

    dist = _get_scipy_dist(fit.name)
    params = fit.params

    if method == "cdf_z":
        try:
            if fit.name == "LogNormal":
                x_adj = max(1e-12, float(x))
                cdf = float(dist.cdf(x_adj, *params))
            else:
                cdf = float(dist.cdf(float(x), *params))
            cdf = min(max(cdf, 1e-12), 1 - 1e-12)
            return float(stats.norm.ppf(cdf))
        except Exception:
            return None

    elif method == "mean_std":
        try:
            if fit.name == "Normal":
                mu, sigma = params
                if sigma <= 0:
                    return None
                return (float(x) - mu) / sigma
            elif fit.name == "LogNormal":
                shape, loc, scale = params
                mu_log = math.log(scale)
                sigma_log = shape
                x_adj = max(1e-12, float(x) - loc)
                return (math.log(x_adj) - mu_log) / sigma_log
            elif fit.name == "Gamma":
                a, loc, scale = params
                mean = a * scale + loc
                std = math.sqrt(a) * scale
                if std <= 0:
                    return None
                return (float(x) - mean) / std
            else:
                return None
        except Exception:
            return None
    else:
        raise ValueError(f"Unknown scoring method: {method}")


def compute_category_scores(
    df: pd.DataFrame,
    best_fits: Dict[str, FitResult],
    points_categories: List[str],
    banger_categories: List[str],
    weights: Optional[Dict[str, float]] = None,
    method: str = "cdf_z",
) -> pd.DataFrame:
    df_out = df.copy()

    available_points = [c for c in points_categories if c in df_out.columns]
    available_banger = [c for c in banger_categories if c in df_out.columns]
    all_cats = available_points + available_banger

    for col in all_cats:
        fit = best_fits.get(col)
        if fit is None:
            print(f"No fit available for category '{col}', cannot score it.")
            continue
        scores: List[Optional[float]] = []
        ser = pd.to_numeric(df_out[col], errors='coerce')
        for val in ser.fillna(np.nan).values:
            scores.append(value_score_from_fit(val, fit, method=method))
        df_out[f"{col}_score"] = pd.Series(scores, index=df_out.index, dtype="float64")

    def mean_across(cols: List[str]) -> pd.Series:
        score_cols = [f"{c}_score" for c in cols if f"{c}_score" in df_out.columns]
        if not score_cols:
            return pd.Series([np.nan] * len(df_out), index=df_out.index)
        return df_out[score_cols].mean(axis=1, skipna=True)

    df_out["PointsScore"] = mean_across(available_points)
    df_out["BangerScore"] = mean_across(available_banger)

    if not all_cats:
        df_out["OverallScore"] = np.nan
        return df_out

    eff_weights: Dict[str, float] = {}
    if weights:
        for c in all_cats:
            eff_weights[c] = float(weights.get(c, 1.0))
    else:
        for c in all_cats:
            eff_weights[c] = 1.0

    score_cols = [f"{c}_score" for c in all_cats if f"{c}_score" in df_out.columns]
    if not score_cols:
        df_out["OverallScore"] = np.nan
        return df_out

    w_arr = np.array([eff_weights[c.replace("_score", "")] for c in score_cols], dtype=float)
    w_arr = np.where(np.isfinite(w_arr), w_arr, 0.0)

    scores_matrix = df_out[score_cols].values
    weights_broadcast = np.tile(w_arr, (scores_matrix.shape[0], 1))
    mask = np.isfinite(scores_matrix)
    weighted_sum = np.nansum(scores_matrix * weights_broadcast * mask, axis=1)
    weight_sum = np.nansum(weights_broadcast * mask, axis=1)
    overall = np.divide(weighted_sum, weight_sum, out=np.full_like(weighted_sum, np.nan, dtype=float), where=weight_sum > 0)
    df_out["OverallScore"] = overall

    return df_out



def scale_scores(df: pd.DataFrame, mode: str = "percentile", columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Scale score columns to a 0–100 presentation range.
    - mode="percentile": assumes inputs are z-scores; maps via Phi(z) * 100.
    - mode="minmax": linear scale per column using observed finite min/max; constant columns map to 50.
    - mode="rank": percentile rank per column in [0,100].
    - mode="none": return df unchanged.

    columns: which columns to scale. By default, all columns ending with "_score" plus
    aggregate columns ["PointsScore", "BangerScore", "OverallScore"] if present.
    """
    if mode is None or mode == "none":
        return df

    df_out = df.copy()

    # Determine target columns
    target_cols: List[str] = []
    if columns is not None:
        target_cols = [c for c in columns if c in df_out.columns]
    else:
        target_cols = [c for c in df_out.columns if c.endswith("_score")]
        for agg in ["PointsScore", "BangerScore", "OverallScore"]:
            if agg in df_out.columns:
                target_cols.append(agg)

    for col in target_cols:
        s = pd.to_numeric(df_out[col], errors="coerce")
        mask = np.isfinite(s.values)
        if not mask.any():
            continue

        if mode == "percentile":
            # Map z-scores to percentile [0,100]
            z = s.values.copy()
            scaled = np.full_like(z, np.nan, dtype=float)
            scaled[mask] = stats.norm.cdf(z[mask]) * 100.0
            df_out[col] = scaled
        elif mode == "minmax":
            finite_vals = s.values[mask]
            vmin = float(np.min(finite_vals))
            vmax = float(np.max(finite_vals))
            if math.isclose(vmin, vmax):
                # Avoid divide-by-zero; map all to 50
                scaled = np.full(s.shape, np.nan, dtype=float)
                scaled[mask] = 50.0
            else:
                scaled = (s.values - vmin) / (vmax - vmin) * 100.0
            df_out[col] = scaled
        elif mode == "rank":
            # Percentile rank in [0,100]
            df_out[col] = s.rank(pct=True, method="average") * 100.0
        else:
            # Unknown mode -> no change for safety
            pass

    return df_out


def apply_pos_group_weighted_overall(
    df: pd.DataFrame,
    weights_map: Optional[Dict[str, tuple]] = None,
    points_col: str = "PointsScore",
    banger_col: str = "BangerScore",
    pos_group_col: str = "pos_group",
    balanced_col: str = "BalancedScore",
    overall_col: str = "OverallScore",
) -> pd.DataFrame:
    """
    Compute a role-aware overall score as a weighted combination of PointsScore and BangerScore.
    - weights_map maps pos_group -> (w_points, w_banger). Supported keys: 'F','D','B','DEFAULT'.
    - If one component is NaN for a row, renormalize to the available component so a valid score is still produced.
    - Writes both `balanced_col` and `overall_col` (set equal by design) and returns a copy.
    """
    if weights_map is None:
        weights_map = {
            'F': (0.6, 0.4),
            'D': (0.4, 0.6),
            'B': (0.5, 0.5),
            'DEFAULT': (0.5, 0.5),
        }

    df_out = df.copy()

    # Prepare arrays
    p = pd.to_numeric(df_out.get(points_col, pd.Series(np.nan, index=df_out.index)), errors='coerce').values
    b = pd.to_numeric(df_out.get(banger_col, pd.Series(np.nan, index=df_out.index)), errors='coerce').values

    # Map pos_group to weights
    groups = df_out.get(pos_group_col, pd.Series(['DEFAULT'] * len(df_out), index=df_out.index)).fillna('DEFAULT').astype(str).values
    wP = np.zeros(len(df_out), dtype=float)
    wB = np.zeros(len(df_out), dtype=float)

    for key, (wp, wb) in weights_map.items():
        mask = (groups == key)
        wP[mask] = float(wp)
        wB[mask] = float(wb)
    # Fill any remaining with DEFAULT
    default_wp, default_wb = weights_map.get('DEFAULT', (0.5, 0.5))
    remaining = (wP == 0) & (wB == 0)
    if remaining.any():
        wP[remaining] = float(default_wp)
        wB[remaining] = float(default_wb)

    # Renormalize when one component is missing per row
    p_mask = np.isfinite(p)
    b_mask = np.isfinite(b)
    denom = (wP * p_mask) + (wB * b_mask)
    numer = (wP * np.where(p_mask, p, 0.0)) + (wB * np.where(b_mask, b, 0.0))
    combined = np.divide(numer, denom, out=np.full_like(numer, np.nan, dtype=float), where=denom > 0)

    df_out[balanced_col] = combined
    df_out[overall_col] = combined
    return df_out
