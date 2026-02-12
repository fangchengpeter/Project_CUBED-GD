#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from typing import Dict, Tuple

import torch
import numpy as np
import matplotlib

# ===== Backend =====
SHOW = False
if not SHOW:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ===== Global style =====
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
})

HERE = os.path.dirname(os.path.abspath(__file__))

# ===== Data layout =====
BASELINE_PATHS: Dict[int, str] = {
    1: "results_trimmed_mean_random_moderate_j1",
    2: "results_trimmed_mean_random_moderate_j2",
    5: "results_trimmed_mean_random_moderate_j5",
    10: "results_trimmed_mean_random_moderate_j10",
}

J5_PATHS_BY_B: Dict[int, str] = {
    0: "results_trimmed_mean_random_moderate_j5_b0",
    1: "results_trimmed_mean_random_moderate_j5_b1",
    2: "results_trimmed_mean_random_moderate_j5_b2",
    4: "results_trimmed_mean_random_moderate_j5_b4",
}

TITLE = "CIFAR10 Moderate Non-IID"
XLABEL = r"Number of Total Communication ($t_c$)"
YLABEL = "Test Accuracy"
SAVE_DPI = 300
SHOW_LEGEND = True

MAX_ROUNDS = 1000
X_DISPLAY_MAX = 100

# ===== Smoothing =====
APPLY_SMOOTH = True
SMOOTH_METHOD = "ema"  # 'ema' | 'ma' | 'savgol'
SMOOTH_WINDOW = 10     # for 'ma' / baseline window size
EMA_SPAN = 40          # larger = smoother
SAVGOL_POLYORDER = 3   # used if method == 'savgol'

# ===== Markers for each curve label =====
MARKERS: Dict[str, str] = {
    # left panel: different local updates (J)
    "J=1": "o",
    "J=2": "s",
    "J=5": "^",
    "J=10": "D",
    # right panel: different Byzantine clients (b)
    "b=0": "o",
    "b=1": "s",
    "b=2": "^",
    "b=4": "D",
}
DEFAULT_MARKER = "o"

def _markevery(n_points: int) -> int:
    return max(1, n_points // 12)

# ===== Utils =====
def safe_slug(s: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _first_array_like_from_dict(d) -> np.ndarray:
    for v in d.values():
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy().astype(float)
        if isinstance(v, (list, tuple, np.ndarray)):
            return np.asarray(v, dtype=float)
    return None

def load_series(folder_rel: str, filename: str = "mean_accuracy.pt"):
    folder = os.path.join(HERE, folder_rel)
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        print(f"[WARN] missing {path}")
        return None
    try:
        data = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[WARN] load failed {path}: {exc}")
        return None

    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(float)
    if isinstance(data, (list, tuple, np.ndarray)):
        return np.asarray(data, dtype=float)
    if isinstance(data, dict):
        arr = _first_array_like_from_dict(data)
        if arr is not None:
            return arr
    print(f"[WARN] unsupported data format in {path}: {type(data)}")
    return None

def _odd_window(cap: int, prefer: int) -> int:
    if cap < 3:
        return max(1, cap)
    w = min(cap, max(3, prefer))
    if w % 2 == 0:
        w -= 1
    if w < 3 and cap >= 3:
        w = 3
    return w

def _smooth_ma(y: np.ndarray, window: int) -> np.ndarray:
    w = max(3, int(window))
    w = min(w, len(y))
    w = min(w, max(5, int(len(y) * 0.25)))
    kernel = np.ones(w, dtype=float) / float(w)
    z = np.convolve(y, kernel, mode="valid")
    pad_left = (len(y) - len(z)) // 2
    pad_right = len(y) - len(z) - pad_left
    return np.pad(z, (pad_left, pad_right), mode="edge")

def _smooth_ema(y: np.ndarray, span: int) -> np.ndarray:
    span = max(2, int(span))
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out

def _smooth_savgol(y: np.ndarray, prefer_win: int, poly: int) -> np.ndarray:
    try:
        cap = len(y) if len(y) % 2 == 1 else len(y) - 1
        prefer = max(prefer_win, 9)
        win = _odd_window(cap, prefer)
        poly = min(max(2, poly), win - 2) if win >= 5 else 2
        max_reasonable = max(5, int(len(y) * 0.35))
        win = _odd_window(cap, min(win, max_reasonable))
        from scipy.signal import savgol_filter  # optional
        return savgol_filter(y, window_length=win, polyorder=poly, mode="interp")
    except Exception:
        # fallback to EMA if scipy not available
        return _smooth_ema(y, span=max(25, EMA_SPAN))

def smooth(y, window=5) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if not APPLY_SMOOTH or len(y) <= 5:
        return y
    m = (SMOOTH_METHOD or "ema").lower()
    if m == "ema":
        return _smooth_ema(y, EMA_SPAN)
    elif m == "savgol":
        return _smooth_savgol(y, prefer_win=window, poly=SAVGOL_POLYORDER)
    else:
        return _smooth_ma(y, window)

def prep_xy(y) -> Tuple[np.ndarray, np.ndarray]:
    if y is None or len(y) == 0:
        return None, None
    n = min(len(y), MAX_ROUNDS)
    y = np.asarray(y[:n], dtype=float)
    xs = np.array([0.0]) if n == 1 else np.linspace(0.0, X_DISPLAY_MAX, n)
    return xs, y

def build_baseline_series():
    out = {}
    for j, folder in sorted(BASELINE_PATHS.items()):
        series = load_series(folder)
        if series is not None:
            out[f"J={j}"] = series
    return out

def build_j5_series_by_b():
    out = {}
    for b, folder in sorted(J5_PATHS_BY_B.items()):
        series = load_series(folder)
        if series is not None:
            out[f"b={b}"] = series
    return out

def plot_panel(ax, curves: Dict[str, np.ndarray], subtitle: str) -> bool:
    any_drawn = False
    prepared = {}
    for label, series in curves.items():
        xs, ys = prep_xy(series)
        if xs is None:
            continue
        ys = smooth(ys, SMOOTH_WINDOW)
        prepared[label] = (xs, ys)

    for label, (xs, ys) in prepared.items():
        ax.plot(
            xs, ys,
            linewidth=2.4,
            alpha=0.95,
            linestyle="-",
            label=label,
            marker=MARKERS.get(label, DEFAULT_MARKER),
            markersize=6,
            markevery=_markevery(len(xs)),
        )
        any_drawn = True

    ax.set_title(subtitle, fontweight="bold", pad=6)
    ax.set_xlim(0, X_DISPLAY_MAX)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 40, 60, 80, 100])
    ax.grid(True, linestyle="--", alpha=0.45)
    return any_drawn

def main():
    # 稍大画布便于阅读与打印
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=SAVE_DPI, sharex=False, sharey=True)

    ax_baseline = axes[0]
    show_baseline = plot_panel(ax_baseline, build_baseline_series(), "different local updates (J)")

    ax_j5 = axes[1]
    show_j5 = plot_panel(ax_j5, build_j5_series_by_b(), "different Byzantine clients (b)")

    # Shared labels
# 让全局坐标轴标签完全居中
    fig.supylabel(YLABEL, fontsize=18, x=0.12, y=0.55, fontweight="bold", ha="center", va="center")
    fig.supxlabel(XLABEL, fontsize=18, x=0.55, y=0.18, fontweight="bold", ha="center", va="center")


    # Legends
    if SHOW_LEGEND and show_baseline:
        handles_j, labels_j = ax_baseline.get_legend_handles_labels()
        ax_baseline.legend(handles_j, labels_j, loc="lower right", frameon=True, fancybox=True)
    if SHOW_LEGEND and show_j5:
        handles_b, labels_b = ax_j5.get_legend_handles_labels()
        ax_j5.legend(handles_b, labels_b, loc="lower right", frameon=True, fancybox=True)

    fig.tight_layout(rect=[0.06, 0.10, 0.98, 0.98])

    out_name = f"{safe_slug(TITLE)}_baseline_vs_j5.png"
    out_path = os.path.join(HERE, out_name)
    fig.savefig(out_path, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()
