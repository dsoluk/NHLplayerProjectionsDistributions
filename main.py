import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


EXCEL_PATH = r"C:\Users\soluk\OneDrive\Documents\FantasyNHL\NatePts.xlsx"
PREFERRED_SHEET_NAME = "NatePts"  # Will fall back to the first sheet if not found

# Category groups for scoring
POINTS_CATEGORIES = ["G", "A", "PPP", "SOG"]
BANGER_CATEGORIES = ["HIT", "BLK", "PIM"]

# Default per-category weights for overall score. If empty or missing keys, equal weights are used.
DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {}


@dataclass
class FitResult:
    name: str
    params: Tuple
    loglik: float
    aic: float
    bic: float
    n: int
    extra: Dict[str, float]


def safe_read_excel(path: str, preferred_sheet: Optional[str] = None) -> pd.DataFrame:
    """
    Attempts to read the Excel file from OneDrive.
    - Tries the preferred sheet name first (if provided), otherwise falls back to the first sheet.
    - Returns a DataFrame.
    """
    try:
        if preferred_sheet is not None:
            try:
                df = pd.read_excel(path, sheet_name=preferred_sheet, engine="openpyxl")
                print(f"Loaded sheet '{preferred_sheet}' from {path}")
                return df
            except Exception as e:
                print(f"Could not open sheet '{preferred_sheet}': {e}. Falling back to the first sheet...")
        # Fall back to first sheet
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
        print(f"Loaded first sheet from {path}")
        return df
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel file at {path}: {e}")




def logpdf_sum(dist, data: np.ndarray, params: Tuple, is_discrete: bool) -> float:
    if is_discrete:
        pmf_vals = dist.pmf(data, *params)
        pmf_vals = np.clip(pmf_vals, 1e-300, 1.0)
        return float(np.sum(np.log(pmf_vals)))
    else:
        pdf_vals = dist.pdf(data, *params)
        pdf_vals = np.clip(pdf_vals, 1e-300, np.inf)
        return float(np.sum(np.log(pdf_vals)))


def compute_ic(loglik: float, k: int, n: int) -> Tuple[float, float]:
    aic = 2 * k - 2 * loglik
    bic = k * math.log(max(n, 1)) - 2 * loglik
    return aic, bic


def choose_best_fit(fits: List[FitResult]) -> Optional[FitResult]:
    if not fits:
        return None
    return sorted(fits, key=lambda fr: fr.aic)[0]


def get_scipy_dist(name: str):
    if name == "Normal":
        return stats.norm
    if name == "LogNormal":
        return stats.lognorm
    if name == "Gamma":
        return stats.gamma
    raise ValueError(f"Unsupported distribution name: {name}")


def value_score_from_fit(x: float, fit: FitResult, method: str = "cdf_z") -> Optional[float]:
    """
    Compute a standardized score for a single value x using the provided fitted distribution.
    Methods:
    - "cdf_z": distribution-aware z via z = Phi^{-1}(F(x)), clipped to avoid infinities.
    - "mean_std": classic z = (x - mean)/std for Normal; for LogNormal uses log-scale; for Gamma uses moment std.
    Returns None if x is NaN or cannot be scored.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None

    dist = get_scipy_dist(fit.name)
    params = fit.params

    if method == "cdf_z":
        try:
            # Handle edge cases for strictly positive dists
            if fit.name == "LogNormal":
                # Ensure x > loc; our fits constrain loc=0
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
                # params: shape (sigma), loc, scale (exp(mu)) when loc=0
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
    """
    Computes per-category standardized scores using best-by-AIC fits and aggregates:
    - Adds <col>_score columns for each category present.
    - Adds PointsScore, BangerScore, and OverallScore columns.
    weights: optional dict of per-category weights for OverallScore (defaults to equal weights for available categories).
    method: "cdf_z" (default) or "mean_std".
    """
    df_out = df.copy()

    available_points = [c for c in points_categories if c in df_out.columns]
    available_banger = [c for c in banger_categories if c in df_out.columns]
    all_cats = available_points + available_banger

    # Create per-category score columns
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

    # Helper to average across available score columns
    def mean_across(cols: List[str]) -> pd.Series:
        score_cols = [f"{c}_score" for c in cols if f"{c}_score" in df_out.columns]
        if not score_cols:
            return pd.Series([np.nan] * len(df_out), index=df_out.index)
        return df_out[score_cols].mean(axis=1, skipna=True)

    df_out["PointsScore"] = mean_across(available_points)
    df_out["BangerScore"] = mean_across(available_banger)

    # Overall weighted average across all categories
    if not all_cats:
        df_out["OverallScore"] = np.nan
        return df_out

    # Determine weights per category
    eff_weights: Dict[str, float] = {}
    if weights:
        for c in all_cats:
            eff_weights[c] = float(weights.get(c, 1.0))
    else:
        for c in all_cats:
            eff_weights[c] = 1.0

    # Build weighted average of per-category scores
    score_cols = [f"{c}_score" for c in all_cats if f"{c}_score" in df_out.columns]
    if not score_cols:
        df_out["OverallScore"] = np.nan
        return df_out

    # Align weights to columns
    w_arr = np.array([eff_weights[c.replace("_score", "")] for c in score_cols], dtype=float)
    w_arr = np.where(np.isfinite(w_arr), w_arr, 0.0)

    # Weighted mean ignoring NaNs per row
    scores_matrix = df_out[score_cols].values
    weights_broadcast = np.tile(w_arr, (scores_matrix.shape[0], 1))
    mask = np.isfinite(scores_matrix)
    weighted_sum = np.nansum(scores_matrix * weights_broadcast * mask, axis=1)
    weight_sum = np.nansum(weights_broadcast * mask, axis=1)
    overall = np.divide(weighted_sum, weight_sum, out=np.full_like(weighted_sum, np.nan, dtype=float), where=weight_sum > 0)
    df_out["OverallScore"] = overall

    return df_out


def fit_normal(x: np.ndarray) -> FitResult:
    mu, sigma = stats.norm.fit(x)
    loglik = logpdf_sum(stats.norm, x, (mu, sigma), is_discrete=False)
    aic, bic = compute_ic(loglik, k=2, n=len(x))
    ks_stat, ks_p = stats.kstest(x, 'norm', args=(mu, sigma))
    return FitResult("Normal", (mu, sigma), loglik, aic, bic, len(x), {"ks_stat": ks_stat, "ks_p": ks_p})


def fit_lognormal(x: np.ndarray) -> Optional[FitResult]:
    x_pos = x[x > 0]
    if len(x_pos) < max(10, int(0.5 * len(x))):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Constrain loc to 0 for identifiability in many practical cases
        shape, loc, scale = stats.lognorm.fit(x_pos, floc=0)
    loglik = logpdf_sum(stats.lognorm, x_pos, (shape, loc, scale), is_discrete=False)
    aic, bic = compute_ic(loglik, k=3, n=len(x_pos))
    ks_stat, ks_p = stats.kstest(x_pos, 'lognorm', args=(shape, loc, scale))
    return FitResult("LogNormal", (shape, loc, scale), loglik, aic, bic, len(x_pos), {"ks_stat": ks_stat, "ks_p": ks_p})


def fit_gamma(x: np.ndarray) -> Optional[FitResult]:
    x_pos = x[x > 0]
    if len(x_pos) < max(10, int(0.5 * len(x))):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, loc, scale = stats.gamma.fit(x_pos, floc=0)
    loglik = logpdf_sum(stats.gamma, x_pos, (a, loc, scale), is_discrete=False)
    aic, bic = compute_ic(loglik, k=3, n=len(x_pos))
    ks_stat, ks_p = stats.kstest(x_pos, 'gamma', args=(a, loc, scale))
    return FitResult("Gamma", (a, loc, scale), loglik, aic, bic, len(x_pos), {"ks_stat": ks_stat, "ks_p": ks_p})




def summarize_results(col_name: str, fits: List[FitResult]) -> None:
    if not fits:
        print(f"Column '{col_name}': No distributions could be fit (insufficient or incompatible data).\n")
        return
    fits_sorted = sorted(fits, key=lambda fr: fr.aic)
    print(f"=== Column: {col_name} (n={fits_sorted[0].n}) ===")
    print("Model\tAIC\tBIC\tLogLik\tExtra")
    for fr in fits_sorted:
        extra_str = ", ".join([f"{k}={v:.4g}" for k, v in fr.extra.items()]) if fr.extra else ""
        print(f"{fr.name}\t{fr.aic:.2f}\t{fr.bic:.2f}\t{fr.loglik:.2f}\t{extra_str}")
    print(f"Best by AIC: {fits_sorted[0].name}")
    print()


def plot_column_with_fits(col_name: str, x: np.ndarray, fits: List[FitResult]) -> None:
    # Organize 3 subplots: Normal, LogNormal, Gamma
    fit_map = {fr.name: fr for fr in fits}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"Fitted distributions for {col_name}")

    # Continuous plotting helper
    def plot_continuous(ax, data, dist, params, title):
        ax.hist(data, bins=min(30, max(10, int(np.sqrt(len(data))))), density=True, alpha=0.6, color='skyblue', edgecolor='k')
        xmin, xmax = np.min(data), np.max(data)
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5
        xs = np.linspace(xmin, xmax, 400)
        ys = dist.pdf(xs, *params)
        ax.plot(xs, ys, 'r-', lw=2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Normal
    ax = axes[0]
    if "Normal" in fit_map:
        fr = fit_map["Normal"]
        plot_continuous(ax, x, stats.norm, fr.params, "Normal")
    else:
        ax.set_visible(False)

    # LogNormal
    ax = axes[1]
    if "LogNormal" in fit_map:
        fr = fit_map["LogNormal"]
        x_pos = x[x > 0]
        plot_continuous(ax, x_pos, stats.lognorm, fr.params, "LogNormal")
    else:
        ax.set_visible(False)

    # Gamma
    ax = axes[2]
    if "Gamma" in fit_map:
        fr = fit_map["Gamma"]
        x_pos = x[x > 0]
        plot_continuous(ax, x_pos, stats.gamma, fr.params, "Gamma")
    else:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def coerce_specific_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    converted = []
    missing = []
    for col in columns:
        if col in df.columns:
            # Only attempt conversion if not already numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Strip common formatting (commas, spaces)
                ser = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
                df[col] = pd.to_numeric(ser, errors='coerce')
                converted.append(col)
        else:
            missing.append(col)
    if converted:
        print(f"Coerced to numeric: {converted}")
    if missing:
        print(f"Columns not found (skipped): {missing}")
    return df


def analyze_dataframe(df: pd.DataFrame, show_plots: bool = True) -> Dict[str, FitResult]:
    # Select numeric columns only
    num_df = df.select_dtypes(include=[np.number])
    best_fit_by_col: Dict[str, FitResult] = {}
    if num_df.shape[1] == 0:
        print("No numeric columns found in the sheet. Nothing to analyze.")
        return best_fit_by_col

    for col in num_df.columns:
        x_raw = num_df[col].dropna().values.astype(float)
        if len(x_raw) < 10:
            print(f"Column '{col}': too few non-NaN observations ({len(x_raw)}), skipping.\n")
            continue

        fits: List[FitResult] = []
        # Normal
        try:
            fits.append(fit_normal(x_raw))
        except Exception as e:
            print(f"Normal fit failed for '{col}': {e}")

        # LogNormal
        try:
            fr = fit_lognormal(x_raw)
            if fr is not None:
                fits.append(fr)
        except Exception as e:
            print(f"LogNormal fit failed for '{col}': {e}")

        # Gamma
        try:
            fr = fit_gamma(x_raw)
            if fr is not None:
                fits.append(fr)
        except Exception as e:
            print(f"Gamma fit failed for '{col}': {e}")

        summarize_results(col, fits)
        best = choose_best_fit(fits)
        if best is not None:
            best_fit_by_col[col] = best
        if show_plots:
            try:
                plot_column_with_fits(col, x_raw, fits)
            except Exception as e:
                print(f"Plotting failed for '{col}': {e}")

    return best_fit_by_col


def main():
    try:
        df = safe_read_excel(EXCEL_PATH, preferred_sheet=PREFERRED_SHEET_NAME)
    except FileNotFoundError:
        print(f"Excel file not found at: {EXCEL_PATH}")
        return
    except Exception as e:
        print(str(e))
        return

    # Ensure new count columns are numeric if present
    df = coerce_specific_columns(df, ["HIT", "BLK", "PIM"])

    print("Data loaded. Columns available:")
    print(list(df.columns))
    print()

    # Analyze and get best fits by column
    best_fits = analyze_dataframe(df, show_plots=True)

    # Prepare weights (can be customized later). If empty, equal weights are used.
    category_weights = DEFAULT_CATEGORY_WEIGHTS

    # Compute scores for requested categories
    if best_fits:
        print("Computing standardized scores using best-by-AIC distributions...")
        scored_df = compute_category_scores(
            df,
            best_fits,
            POINTS_CATEGORIES,
            BANGER_CATEGORIES,
            weights=category_weights,
            method="cdf_z",
        )

        # Print summary of chosen best distributions for relevant categories
        relevant_cols = [c for c in POINTS_CATEGORIES + BANGER_CATEGORIES if c in df.columns]
        print("\nBest-by-AIC distributions for scoring:")
        for col in relevant_cols:
            fit = best_fits.get(col)
            if fit:
                print(f"  {col}: {fit.name}")
            else:
                print(f"  {col}: no fit available")

        # Show a preview of the scores
        score_columns = [f"{c}_score" for c in relevant_cols if f"{c}_score" in scored_df.columns] + [
            "PointsScore", "BangerScore", "OverallScore"
        ]
        existing_score_columns = [c for c in score_columns if c in scored_df.columns]
        with pd.option_context('display.max_columns', None, 'display.width', 200):
            print("\nScore preview (first 10 rows):")
            print(scored_df[existing_score_columns].head(10))

        # Save full scored output to CSV at requested path
        try:
            output_path = r"C:\Users\soluk\Downloads\NateProj2526_scores.csv"
            scored_df.to_csv(output_path, index=False)
            print(f"\nScored output saved to: {output_path}")
        except Exception as e:
            print(f"\nFailed to save scored output to CSV: {e}")
    else:
        print("No fits were produced; skipping scoring.")


if __name__ == '__main__':
    main()
