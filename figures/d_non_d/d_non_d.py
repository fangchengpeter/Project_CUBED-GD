#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import matplotlib

# ===== 显示后端 =====
SHOW = False
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ===== 全局样式 =====
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20
})

# ===== 文案与参数 =====
XLABEL = "Number of Total Communication ($t_c$)"
YLABEL = "Test Accuracy"
SAVE_DPI = 300
Y_LIM = (0.0, 100.0)
SMOOTH_WINDOW = 5          # =1 可关闭平滑
DEBUG = True               # 打开调试

# ===== 数据路径 =====
# CIFAR-10 (non-defense)
cifar10_b0_j5 = "cifar10/results_none_random_b0_j5"
cifar10_b1_j5 = "cifar10/results_none_random_b1_j5"
cifar10_b2_j5 = "cifar10/results_none_random_b2_j5"
cifar10_b4_j5 = "cifar10/results_none_random_b4_j5"
# CIFAR-10 (defense)
cifar10_trimmed_mean_b1 = "cifar10/results_trimmed_mean_random_cifar_b1_j5"
cifar10_trimmed_mean_b2 = "cifar10/results_trimmed_mean_random_cifar_b2_j5"
cifar10_trimmed_mean_b4 = "cifar10/results_trimmed_mean_random_cifar_b4_j5"

# MNIST (non-defense)
mnist_b1_j5 = "mnist/results_none_random_b1_j5"
mnist_b2_j5 = "mnist/results_none_random_b2_j5"
mnist_b4_j5 = "mnist/results_none_random_b4_j5"
mnist_b8_j5 = "mnist/results_none_random_b8_j5"
# MNIST (defense)
mnist_b1_j5_trimmed = "mnist/results_trimmed_mean_random_b1_j5"
mnist_b2_j5_trimmed = "mnist/results_trimmed_mean_random_b2_j5"
mnist_b4_j5_trimmed = "mnist/results_trimmed_mean_random_b4_j5"
mnist_b8_j5_trimmed = "mnist/results_trimmed_mean_random_b8_j5"

# ===== 颜色 & 线型映射 =====
COLOR_MAP = {"b1": "tab:blue", "b2": "tab:orange", "b4": "tab:green", "b8": "tab:red"}
LINESTYLE = {"defense": "-", "none": "--"}   # 实线=defense, 虚线=none

# ===== 工具函数 =====
def safe_load_series(folder, filename="mean_accuracy.pt"):
    """加载序列并转 numpy；失败返回空数组并打印原因。"""
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        if DEBUG: print(f"[WARN] Missing file: {path}")
        return np.array([], dtype=float)
    try:
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        if DEBUG: print(f"[ERR ] torch.load failed: {path} -> {e}")
        return np.array([], dtype=float)

    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy().astype(float)
    elif isinstance(data, (list, tuple, np.ndarray)):
        arr = np.asarray(data, dtype=float)
    elif hasattr(data, "values"):
        for v in data.values():
            if isinstance(v, torch.Tensor):
                arr = v.detach().cpu().numpy().astype(float); break
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v, dtype=float); break
        else:
            if DEBUG: print(f"[ERR ] Unsupported dict-like in {path}")
            return np.array([], dtype=float)
    else:
        if DEBUG: print(f"[ERR ] Unsupported type {type(data)} in {path}")
        return np.array([], dtype=float)

    if arr.size == 0 or not np.isfinite(arr).any():
        if DEBUG: print(f"[WARN] Empty or non-finite series: {path}")
        return np.array([], dtype=float)
    return arr

def smooth(x, window=5):
    """居中滑动平均 + 边界反射，长度不变，稳妥平滑。"""
    x = np.asarray(x, dtype=float)
    if x.size <= 2 or window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    window = min(window, max(3, x.size // 4))
    pad = window // 2
    x_pad = np.pad(x, pad_width=pad, mode="reflect")
    kernel = np.ones(window, dtype=float) / float(window)
    y = np.convolve(x_pad, kernel, mode="valid")
    return y

def debug_series(tag, curves):
    if not DEBUG: return
    print(f"\n[DEBUG] {tag}")
    for c in curves:
        y = np.asarray(c["y"], dtype=float)
        if y.size == 0:
            print(f" - SKIP  b={c['b']:>2s} mode={c['mode']:<8s}  len=0")
            continue
        mn = np.nanmin(y); mx = np.nanmax(y)
        print(f" - OK    b={c['b']:>2s} mode={c['mode']:<8s}  len={len(y):4d}  min={mn:.4f}  max={mx:.4f}  head={y[:5]}")

# ===== 绘图函数 =====
def plot_panel(ax, title, curves):
    plotted = 0
    for c in curves:
        y = np.asarray(c["y"], dtype=float)
        if y.size == 0:
            continue
        y = smooth(y, SMOOTH_WINDOW)
        xs = np.arange(len(y))
        color = COLOR_MAP.get(c["b"], "gray")
        ax.plot(xs, y,
                color=color,
                linestyle=LINESTYLE[c["mode"]],
                linewidth=2.4,
                alpha=0.98)
        plotted += 1

    if DEBUG:
        print(f"[INFO ] Plot '{title}': plotted {plotted} series.")

    ax.set_title(title, fontweight="bold")
    ax.set_ylim(*Y_LIM)
    ax.grid(True, linestyle="--", alpha=0.5)

def two_row_legend(fig, present_bs):
    """
    图例结构：
      上行：CUBED-GD (—)   [蓝 橙 绿 红 对应 b=1,2,4,8]
      下行：Multi local updates DGD (--) [蓝 橙 绿 红 对应 b=1,2,4,8]
    """
    bs = [b for b in ["b1", "b2", "b4", "b8"] if b in present_bs]

    # 第一行：CUBED-GD（defense，实线）
    handles_row1 = [Line2D([0], [0], color=COLOR_MAP[b], lw=3, linestyle="-") for b in bs]
    labels_row1 = [f"b={b[1:]}" for b in bs]

    # 第二行：DGD（none，虚线）
    handles_row2 = [Line2D([0], [0], color=COLOR_MAP[b], lw=3, linestyle="--") for b in bs]
    labels_row2 = [f"b={b[1:]}" for b in bs]

    # 第一行：CUBED-GD (—)
    title1 = Line2D([], [], color='black', lw=0, label="CUBED-GD (—)")
    leg1 = fig.legend(
        handles=[title1] + handles_row1,
        labels=["CUBED-GD (—)"] + labels_row1,
        loc="upper center",
        ncol=len(bs) + 1,
        frameon=True,
        fancybox=True,
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=0.8,
        handlelength=2.0,
        handletextpad=0.5
    )
    leg1.get_frame().set_alpha(0.9)

    # 第二行：DGD (--)
    title2 = Line2D([], [], color='black', lw=0, label="Multi local updates DGD (--)")
    leg2 = fig.legend(
        handles=[title2] + handles_row2,
        labels=["DGD with multi-step local updates (--)"] + labels_row2,
        loc="upper center",
        ncol=len(bs) + 1,
        frameon=True,
        fancybox=True,
        bbox_to_anchor=(0.5, 0.94),
        columnspacing=0.5,
        handlelength=2.0,
        handletextpad=0.5
    )
    leg2.get_frame().set_alpha(0.9)


def plot_all(mnist_curves, cifar_curves):
    debug_series("MNIST", mnist_curves)
    debug_series("CIFAR-10", cifar_curves)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=SAVE_DPI, sharey=True)

    plot_panel(axes[0], "MNIST", mnist_curves)
    plot_panel(axes[1], "CIFAR-10", cifar_curves)

    # 出现过的 b（用于图例），且不含 b0
    present_bs = set(
        [c["b"] for c in mnist_curves if len(c["y"]) > 0 and c["b"] != "b0"] +
        [c["b"] for c in cifar_curves if len(c["y"]) > 0 and c["b"] != "b0"]
    )
    two_row_legend(fig, present_bs)

    fig.supxlabel(XLABEL, fontsize=18, y=0.04,fontweight="bold")
    fig.supylabel(YLABEL, fontsize=18, x=0.05, fontweight="bold")
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.10, right=0.95, wspace=0.05)

    out = "CUBED-GD_vs_Byzantine.png"
    fig.savefig(out, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close(fig)
    print(f"\nSaved: {out}")

# ===== 主程序 =====
def main():
    # MNIST：b1/b2/b4/b8 的 non + defense
    mnist_curves = [
        {"y": safe_load_series(mnist_b1_j5),         "b": "b1", "mode": "none"},
        {"y": safe_load_series(mnist_b2_j5),         "b": "b2", "mode": "none"},
        {"y": safe_load_series(mnist_b4_j5),         "b": "b4", "mode": "none"},
        {"y": safe_load_series(mnist_b8_j5),         "b": "b8", "mode": "none"},
        {"y": safe_load_series(mnist_b1_j5_trimmed), "b": "b1", "mode": "defense"},
        {"y": safe_load_series(mnist_b2_j5_trimmed), "b": "b2", "mode": "defense"},
        {"y": safe_load_series(mnist_b4_j5_trimmed), "b": "b4", "mode": "defense"},
        {"y": safe_load_series(mnist_b8_j5_trimmed), "b": "b8", "mode": "defense"},
    ]
    # CIFAR-10：b0 only non；b1/b2/b4 的 non + defense
    cifar_curves = [
        {"y": safe_load_series(cifar10_b0_j5),             "b": "b0", "mode": "none"},
        {"y": safe_load_series(cifar10_b1_j5),             "b": "b1", "mode": "none"},
        {"y": safe_load_series(cifar10_b2_j5),             "b": "b2", "mode": "none"},
        {"y": safe_load_series(cifar10_b4_j5),             "b": "b4", "mode": "none"},
        {"y": safe_load_series(cifar10_trimmed_mean_b1),   "b": "b1", "mode": "defense"},
        {"y": safe_load_series(cifar10_trimmed_mean_b2),   "b": "b2", "mode": "defense"},
        {"y": safe_load_series(cifar10_trimmed_mean_b4),   "b": "b4", "mode": "defense"},
    ]

    # 过滤空序列 + 过滤 b0（不绘制/不进图例）
    mnist_curves = [c for c in mnist_curves if len(c["y"]) > 0 and c["b"] != "b0"]
    cifar_curves = [c for c in cifar_curves if len(c["y"]) > 0 and c["b"] != "b0"]

    plot_all(mnist_curves, cifar_curves)

if __name__ == "__main__":
    main()
