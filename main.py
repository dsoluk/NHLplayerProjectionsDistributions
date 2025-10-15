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


def analyze_dataframe(df: pd.DataFrame, show_plots: bool = True) -> None:
    # Select numeric columns only
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        print("No numeric columns found in the sheet. Nothing to analyze.")
        return

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
        if show_plots:
            try:
                plot_column_with_fits(col, x_raw, fits)
            except Exception as e:
                print(f"Plotting failed for '{col}': {e}")


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

    analyze_dataframe(df, show_plots=True)


if __name__ == '__main__':
    main()
