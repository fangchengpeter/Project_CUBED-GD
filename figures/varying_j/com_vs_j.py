#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import matplotlib

SHOW = False
if not SHOW:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ===== 全局字体放大 =====
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20
})

# ===== 路径配置 =====
# CIFAR-10
cifar10_j1 = "results_j1_baseline"
cifar10_j2 = "results_j2_baseline"
cifar10_j5 = "results_j5_baseline"
cifar10_j10 = "results_j10_baseline"

# MNIST
mnist_j1 = "results_trimmed_mean_mnist_j1_baseline"
mnist_j2 = "results_trimmed_mean_mnist_j2_baseline"
mnist_j5 = "results_trimmed_mean_mnist_j5_baseline"
mnist_j10 = "results_trimmed_mean_mnist_j10_baseline"

# ===== 可调参数 =====
# TITLE = "CUBED-GD vs Communication Cost"
XLABEL = "Number of Total Communication ($t_c$)"
YLABEL = "Test Accuracy"
AUTO_YLIM = True
ASSUME_PERCENT_IF_GT1 = True
APPLY_SMOOTH = True
SMOOTH_WINDOW = 10
SAVE_DPI = 300
SHOW_LEGEND = True   # ← 如果不要图例，改成 False
LEFT_LEGEND_LOC = "lower right"
RIGHT_LEGEND_LOC = "lower right"

# ===== 工具函数 =====
def load_series(folder, filename="mean_accuracy.pt"):
    path = os.path.join(folder, filename)
    data = torch.load(path, map_location="cpu")
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
    raise ValueError(f"Unsupported data format in {path}: {type(data)}")

def smooth(x, window=5):
    x = np.asarray(x, dtype=float)
    if not APPLY_SMOOTH or window <= 1 or window > len(x):
        return x
    # 保证奇数窗口，中心对齐
    if window % 2 == 0:
        window += 1
    pad = window // 2
    x_pad = np.pad(x, pad_width=pad, mode="reflect")
    kernel = np.ones(window, dtype=float) / window
    y = np.convolve(x_pad, kernel, mode="valid")  # 与原长度相同
    return y



def align_series(series_dict):
    min_len = min(len(v) for v in series_dict.values() if len(v) > 0)
    xs = np.arange(min_len)
    return {k: np.asarray(v[:min_len], dtype=float) for k, v in series_dict.items()}, xs


def plot_panel(ax, title_suffix, series_dict, limit_left=False, show_legend=False, legend_loc="best"):
    series_dict, xs = align_series(series_dict)

    # 如果是 MNIST 左图 → 限制前 1000
    if limit_left:
        xs = xs[:1000]
        series_dict = {k: v[:1000] for k, v in series_dict.items()}

    ymin, ymax = np.inf, -np.inf

    # linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    linestyles = ["-"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for i, (name, y) in enumerate(series_dict.items()):
        y_plot = smooth(y, SMOOTH_WINDOW)
        ymin = min(ymin, float(np.nanmin(y_plot)))
        ymax = max(ymax, float(np.nanmax(y_plot)))

        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]

        # ✅ 将 j1 → J = 1
        if name.lower().startswith("j"):
            try:
                j_val = int(name[1:])
                display_name = rf"$J = {j_val}$"
            except ValueError:
                display_name = name
        else:
            display_name = name

        MARK_EVERY = max(1, len(xs) // 20)
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]

        # --- 第一层：纯线（平滑连贯） ---
        ax.plot(
            xs, y_plot,
            linestyle=linestyle,
            linewidth=2.0,
            color=f"C{i}",  # 使用 matplotlib 默认调色板一致色
            alpha=0.9,
        )

        # --- 第二层：只打点 ---
        ax.plot(
            xs, y_plot,
            linestyle="none",
            marker=marker,
            markevery=MARK_EVERY,
            markersize=6,
            color=f"C{i}",
            label=display_name,
        )



    ax.set_title(f"{title_suffix}", fontweight="bold")
    if AUTO_YLIM and np.isfinite(ymin) and np.isfinite(ymax):
        span = ymax - ymin
        ax.set_ylim(ymin - 0.07 * span, ymax + 0.07 * span)
    ax.grid(True, linestyle="--", alpha=0.5)
    if show_legend:
        legend = ax.legend(loc=legend_loc, frameon=True)
        if legend and legend.get_frame():
            legend.get_frame().set_alpha(0.9)


def plot_all(left_series, right_series, left_title, right_title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=SAVE_DPI, sharex=False, sharey=True)

    plot_panel(
        axes[0],
        left_title,
        left_series,
        limit_left=True,
        show_legend=False,
        legend_loc=LEFT_LEGEND_LOC,
    )
    plot_panel(
        axes[1],
        right_title,
        right_series,
        limit_left=False,
        show_legend=SHOW_LEGEND,
        legend_loc=RIGHT_LEGEND_LOC,
    )

    # 大标题
    # fig.suptitle(TITLE, fontsize=22, fontweight="bold", y=1.05)

    # 共享图例（可开关）
    # 全局坐标轴标签
    fig.supxlabel(XLABEL, fontsize=18, fontweight="bold")
    fig.supylabel(YLABEL, fontsize=18, fontweight="bold")

    # 收紧空白
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.10, right=0.95, wspace=0.08)

    fig.savefig("CUBED-GD vs Communication Cost.png", bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)

def main():
    # 正确对应：MNIST 用 mnist_* 路径；CIFAR10 用 cifar10_* 路径
    MNIST = {
        "j1":  load_series(mnist_j1),
        "j2":  load_series(mnist_j2),
        "j5":  load_series(mnist_j5),
        "j10": load_series(mnist_j10),
    }
    CIFAR10 = {
        "j1":  load_series(cifar10_j1),
        "j2":  load_series(cifar10_j2),
        "j5":  load_series(cifar10_j5),
        "j10": load_series(cifar10_j10),
    }

    # 画图：左 MNIST，右 CIFAR-10
    plot_all(MNIST, CIFAR10, "MNIST", "CIFAR-10")
    print("\nSaved file: COMBRIDGE vs Communication Cost.png")

if __name__ == "__main__":
    main()
