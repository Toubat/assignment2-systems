"""Chart helpers for the benchmark scripts.

matplotlib is imported *lazily* inside each function. The benchmark modules that
call these are also imported inside Modal containers (which don't have
matplotlib) to load the remote function, so importing this module must never
require matplotlib. Charts are only rendered locally, in the local_entrypoint,
after results come back from the GPU.
"""

from pathlib import Path

# Repo-root /charts (this file lives at <repo>/cs336_systems/plots.py).
CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts"

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def _save(fig, filename: str) -> Path:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"wrote chart {path}")
    return path


def _draw_lines(ax, series: dict, logx: bool, logy: bool):
    for i, (label, (xs, ys)) in enumerate(series.items()):
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pts:
            continue
        xs2, ys2 = zip(*pts)
        style = "--" if "ref" in label or "recursive" in label else "-"
        ax.plot(xs2, ys2, style, marker="o", ms=3, label=label, color=_COLORS[i % len(_COLORS)])
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8)


def line_chart(filename, title, xlabel, ylabel, series, logx=False, logy=False) -> Path:
    """series: {label -> (xs, ys)}; ys entries that are None are skipped."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _draw_lines(ax, series, logx, logy)
    return _save(fig, filename)


def grid_line_chart(filename, suptitle, panels, ncols=2, logx=False, logy=False) -> Path:
    """panels: list of {title, xlabel, ylabel, series}."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows), squeeze=False)
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx >= n:
            ax.axis("off")
            continue
        p = panels[idx]
        ax.set_title(p["title"])
        ax.set_xlabel(p["xlabel"])
        ax.set_ylabel(p["ylabel"])
        _draw_lines(ax, p["series"], logx, logy)
    fig.suptitle(suptitle, fontsize=13)
    return _save(fig, filename)


def grouped_bar_chart(filename, title, xlabel, ylabel, categories, groups) -> Path:
    """categories: x-axis labels; groups: {label -> [value per category]} (None -> 0)."""
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(categories)), 5))
    x = np.arange(len(categories))
    n = len(groups)
    width = 0.8 / max(n, 1)
    for i, (label, vals) in enumerate(groups.items()):
        heights = [0.0 if v is None else v for v in vals]
        ax.bar(x + (i - (n - 1) / 2) * width, heights, width, label=label, color=_COLORS[i % len(_COLORS)])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    return _save(fig, filename)
