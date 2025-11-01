import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FitResult:
    name: str
    params: Tuple
    loglik: float
    aic: float
    bic: float
    n: int
    extra: Dict[str, float]


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


def choose_best_fit(fits: List['FitResult']) -> Optional['FitResult']:
    if not fits:
        return None
    return sorted(fits, key=lambda fr: fr.aic)[0]


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
        # Constrain loc to 0 for identifiability
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


def analyze_dataframe(df: pd.DataFrame, show_plots: bool = True, only_columns: Optional[List[str]] = None) -> Dict[str, FitResult]:
    # Select numeric columns only
    num_df = df.select_dtypes(include=[np.number])
    # If a column whitelist is provided, intersect with it
    if only_columns is not None:
        only_set = {c for c in only_columns}
        keep_cols = [c for c in num_df.columns if c in only_set]
        num_df = num_df[keep_cols]
        if num_df.shape[1] == 0:
            print("No numeric columns (from the requested set) found to analyze.")
            return {}

    best_fit_by_col: Dict[str, FitResult] = {}
    if num_df.shape[1] == 0:
        print("No numeric columns found in the sheet. Nothing to analyze.")
        return best_fit_by_col

    for col in num_df.columns:
        x_raw = num_df[col].dropna().values.astype(float)
        if len(x_raw) < 5:
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
                from nhlproj.analysis.plotting import plot_column_with_fits
                plot_column_with_fits(col, x_raw, fits)
            except Exception as e:
                print(f"Plotting failed for '{col}': {e}")

    return best_fit_by_col
