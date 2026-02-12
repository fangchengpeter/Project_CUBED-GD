#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import torch
import numpy as np
import matplotlib

SHOW = False
if not SHOW:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ===== 全局字体放大 =====
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 22
})

# ===== 路径配置（保持你的路径不变） =====
# Moderate Non-IID
moderate_j1_b0 = "results_trimmed_mean_random_moderate_j1_b0"
moderate_j1_b1 = "results_trimmed_mean_random_moderate_j1_b1"
moderate_j1_b2 = "results_trimmed_mean_random_moderate_j1_b2"
moderate_j1_b4 = "results_trimmed_mean_random_moderate_j1_b4"

moderate_j2_b0 = "results_trimmed_mean_random_moderate_j2_b0"
moderate_j2_b1 = "results_trimmed_mean_random_moderate_j2_b1"
moderate_j2_b2 = "results_trimmed_mean_random_moderate_j2_b2"
moderate_j2_b4 = "results_trimmed_mean_random_moderate_j2_b4"

moderate_j5_b0 = "results_trimmed_mean_random_moderate_j5_b0"
moderate_j5_b1 = "results_trimmed_mean_random_moderate_j5_b1"
moderate_j5_b2 = "results_trimmed_mean_random_moderate_j5_b2"
moderate_j5_b4 = "results_trimmed_mean_random_moderate_j5_b4"

moderate_j10_b0 = "results_trimmed_mean_random_moderate_j10_b0"
moderate_j10_b1 = "results_trimmed_mean_random_moderate_j10_b1"
moderate_j10_b2 = "results_trimmed_mean_random_moderate_j10_b2"
moderate_j10_b4 = "results_trimmed_mean_random_moderate_j10_b4"

# Extreme Non-IID
extreme_j1_b0 = "non_iid_trimmed_mean_different_b_j1/results_trimmed_mean_random_b0"
extreme_j1_b1 = "non_iid_trimmed_mean_different_b_j1/results_trimmed_mean_random_b1"
extreme_j1_b2 = "non_iid_trimmed_mean_different_b_j1/results_trimmed_mean_random_b2"
extreme_j1_b4 = "non_iid_trimmed_mean_different_b_j1/results_trimmed_mean_random_b4"

extreme_j2_b0 = "non_iid_trimmed_mean_different_b_j2/results_trimmed_mean_random_b0"
extreme_j2_b1 = "non_iid_trimmed_mean_different_b_j2/results_trimmed_mean_random_b1"
extreme_j2_b2 = "non_iid_trimmed_mean_different_b_j2/results_trimmed_mean_random_b2"
extreme_j2_b4 = "non_iid_trimmed_mean_different_b_j2/results_trimmed_mean_random_b4"

extreme_j5_b0 = "non_iid_trimmed_mean_different_b_j5/results_trimmed_mean_random_b0"
extreme_j5_b1 = "non_iid_trimmed_mean_different_b_j5/results_trimmed_mean_random_b1"
extreme_j5_b2 = "non_iid_trimmed_mean_different_b_j5/results_trimmed_mean_random_b2"
extreme_j5_b4 = "non_iid_trimmed_mean_different_b_j5/results_trimmed_mean_random_b4"

extreme_j10_b0 = "non_iid_trimmed_mean_different_b_j10/results_trimmed_mean_random_b0"
extreme_j10_b1 = "non_iid_trimmed_mean_different_b_j10/results_trimmed_mean_random_b1"
extreme_j10_b2 = "non_iid_trimmed_mean_different_b_j10/results_trimmed_mean_random_b2"
extreme_j10_b4 = "non_iid_trimmed_mean_different_b_j10/results_trimmed_mean_random_b4"

MODERATE_PATHS = {
    0: {1: moderate_j1_b0, 2: moderate_j2_b0, 5: moderate_j5_b0, 10: moderate_j10_b0},
    1: {1: moderate_j1_b1, 2: moderate_j2_b1, 5: moderate_j5_b1, 10: moderate_j10_b1},
    2: {1: moderate_j1_b2, 2: moderate_j2_b2, 5: moderate_j5_b2, 10: moderate_j10_b2},
    4: {1: moderate_j1_b4, 2: moderate_j2_b4, 5: moderate_j5_b4, 10: moderate_j10_b4},
}

EXTREME_PATHS = {
    0: {1: extreme_j1_b0, 2: extreme_j2_b0, 5: extreme_j5_b0, 10: extreme_j10_b0},
    1: {1: extreme_j1_b1, 2: extreme_j2_b1, 5: extreme_j5_b1, 10: extreme_j10_b1},
    2: {1: extreme_j1_b2, 2: extreme_j2_b2, 5: extreme_j5_b2, 10: extreme_j10_b2},
    4: {1: extreme_j1_b4, 2: extreme_j2_b4, 5: extreme_j5_b4, 10: extreme_j10_b4},
}

# ===== 配置 =====
TITLE = "CUBED-GD vs Non-IID"
XLABEL = "Number of Total Communication ($t_c$)"

YLABEL = "Test Accuracy"
APPLY_SMOOTH = True
SMOOTH_WINDOW = 5
SAVE_DPI = 300
SHOW_LEGEND = True

# —— 关键：统一画“前 1000 轮”，并把横坐标映射为 0–100（百分比） ——
MAX_ROUNDS = 1000         # 截断到 1000
X_DISPLAY_MAX = 1000       # 显示为 0–100

# ===== 工具函数 =====
def safe_slug(s: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_series(folder, filename="mean_accuracy.pt"):
    path = os.path.join(folder, filename)
    try:
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[WARN] Cannot load {path}: {e}")
        return None

    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(float)
    if isinstance(data, (list, tuple, np.ndarray)):
        return np.asarray(data, dtype=float)
    if hasattr(data, "values"):
        for v in data.values():
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy().astype(float)
            if isinstance(v, (list, tuple, np.ndarray)):
                return np.asarray(v, dtype=float)
    print(f"[WARN] Unsupported data format in {path}: {type(data)}")
    return None

def smooth(y, window=5):
    y = np.asarray(y, dtype=float)
    if not APPLY_SMOOTH or window <= 1 or window > len(y):
        return y
    kernel = np.ones(window, dtype=float) / float(window)
    z = np.convolve(y, kernel, mode="valid")
    pad_left = (len(y) - len(z)) // 2
    pad_right = len(y) - len(z) - pad_left
    return np.pad(z, (pad_left, pad_right), mode="edge")

def prep_xy(y):
    """截断到前 MAX_ROUNDS，并把 x 转成 0–100 百分比坐标。"""
    if y is None or len(y) == 0:
        return None, None
    n = min(len(y), MAX_ROUNDS)
    y = np.asarray(y[:n], dtype=float)
    if n == 1:
        xs = np.array([0.0])
    else:
        xs = np.linspace(0.0, X_DISPLAY_MAX, n)  # 0..100
    return xs, y

def plot_panel(ax, curves_dict, subtitle=None, show_legend=False, legend_loc="best"):
    """Plot provided curves on ax; returns True if any curve was drawn."""
    any_drawn = False
    prepared = {}

    # 为不同 b 指定线型/marker（可按喜好调整）
    style_map = {
        "b=0": ("-",  "o"),   # 实线 + 圆
        "b=1": ("--", "s"),   # 虚线 + 方
        "b=2": ("-.", "^"),   # 点划线 + 三角
        "b=4": (":",  "D"),   # 点线 + 菱形
    }

    for label, series in curves_dict.items():
        xs, ys = prep_xy(series)
        if xs is None:
            continue
        ys = smooth(ys, SMOOTH_WINDOW)
        prepared[label] = (xs, ys)

    for label, (xs, ys) in prepared.items():
        linestyle, marker = style_map.get(label, ("-", None))
        ax.plot(
            xs, ys,
            linewidth=2.4,
            alpha=0.95,
            linestyle=linestyle,
            marker=marker,
            markevery=max(1, len(xs)//20),  # 隔一段打一个点，更干净
            markersize=5,
            label=label,
        )
        any_drawn = True

    if subtitle:
        ax.text(0.04, 0.04, subtitle, transform=ax.transAxes,
                fontweight="bold", fontsize=18, va="bottom", ha="left")

    ax.set_xlim(0, X_DISPLAY_MAX)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
    ax.tick_params(axis="x", labelsize=14, width=1.2, labelcolor="black")

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=14, width=1.2, labelcolor="black")

    ax.grid(True, linestyle="--", alpha=0.45)

    if show_legend and prepared:
        legend = ax.legend(loc=legend_loc, frameon=True, fancybox=True)
        if legend and legend.get_frame():
            legend.get_frame().set_alpha(0.9)

    return any_drawn


def build_series_moderate_for_j(j):
    series = {}
    for b in sorted(MODERATE_PATHS.keys()):
        path = MODERATE_PATHS[b].get(j)
        if path:
            series[f"b={b}"] = load_series(path)
    return series


def build_series_extreme_for_j(j):
    series = {}
    for b in sorted(EXTREME_PATHS.keys()):
        path = EXTREME_PATHS[b].get(j)
        if path:
            series[f"b={b}"] = load_series(path)
    return series

def main():
    # 固定列数为不同 j，对应每列展示不同 b
    if not MODERATE_PATHS:
        raise RuntimeError("MODERATE_PATHS is empty; nothing to plot.")
    js = sorted({j for mapping in MODERATE_PATHS.values() for j in mapping})
    if not js:
        raise RuntimeError("No j values found in MODERATE_PATHS.")

    # 2x4 布局；把画布下移（rect 上边留更大空间）
    fig, axes = plt.subplots(2, len(js), figsize=(22, 9.8), dpi=SAVE_DPI, sharex=False, sharey=True)

    # 上排：Moderate；下排：Extreme
    for col, j in enumerate(js):
        ax_m = axes[0, col]
        plot_panel(ax_m,
                   build_series_moderate_for_j(j),
                   subtitle=f"J={j}",
                   show_legend=False)

        ax_e = axes[1, col]
        plot_panel(ax_e,
                   build_series_extreme_for_j(j),
                   subtitle=f"J={j}",
                   show_legend=SHOW_LEGEND and col == len(js) - 1,
                   legend_loc="lower right")

    # 只保留一条全局 X/Y label（一个共享 y）
    fig.supylabel(YLABEL, fontsize=20, x=0.05, fontweight="bold")
    fig.supxlabel(XLABEL, fontsize=20, y=0.1, fontweight="bold")
    # fig.text(0.02, 0.75, "Moderate Non-IID", fontsize=18, fontweight="bold",
    #          rotation=90, va="center", ha="center")
    # fig.text(0.02, 0.25, "Extreme Non-IID", fontsize=18, fontweight="bold",
    #          rotation=90, va="center", ha="center")
    # fig.suptitle(TITLE, fontsize=22, fontweight="bold", y=0.98)

    # 下移子图区域，给标题+图例留空间
    fig.tight_layout(rect=[0.03, 0.06, 0.97, 0.95])

    out_name = f"{safe_slug(TITLE)}_8panels_0to100x.png"
    fig.savefig(out_name, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)
    print(f"[OK] Saved: {out_name}")

if __name__ == "__main__":
    main()
