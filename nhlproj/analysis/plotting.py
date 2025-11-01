from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from nhlproj.analysis.fitting import FitResult


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
