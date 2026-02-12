import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------- helpers ----------
def to_1d_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().float().view(-1)
    if isinstance(x, (list, tuple, np.ndarray)):
        return torch.tensor(x, dtype=torch.float32).view(-1)
    if isinstance(x, dict):
        parts = []
        for v in x.values():
            if isinstance(v, torch.Tensor):
                parts.append(v.detach().float().view(-1))
            elif isinstance(v, (list, tuple, np.ndarray)):
                parts.append(torch.tensor(v, dtype=torch.float32).view(-1))
        if parts:
            return torch.cat(parts)
    raise ValueError(f"Unsupported parameter format: {type(x)}")

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

def first_reach_threshold(acc, thr=90.0):
    if acc is None:
        return -1
    for i, v in enumerate(acc):
        if v >= thr:
            return i
    return -1

# ---------- groups ----------
mean_groups = {
    "mnist1": {
        1: "results_trimmed_mean_random_mnist_mean0_j1", # mean 0 std 1
        2: "results_trimmed_mean_random_mnist_mean0_j2",
        5: "results_trimmed_mean_random_mnist_mean0_j5",
        10: "results_trimmed_mean_random_mnist_mean0_j10"
    },
    "mnist2": {
        1: "results_trimmed_mean_random_mnist_mean3_j1", # mean 3 std 1
        2: "results_trimmed_mean_random_mnist_mean3_j2",
        5: "results_trimmed_mean_random_mnist_mean3_j5",
        10: "results_trimmed_mean_random_mnist_mean3_j10",
        # 20: "results_trimmed_mean_random_mnist_mean3_j20",
    },
    "mnist3": {
        1: "results_trimmed_mean_random_mnist_mean5_j1", # mean 5 std 2
        2: "results_trimmed_mean_random_mnist_mean5_j2",
        5: "results_trimmed_mean_random_mnist_mean5_j5",
        10: "results_trimmed_mean_random_mnist_mean5_j10",
        # 20: "results_trimmed_mean_random_mnist_mean5_j20",
    },
    "cifar10_1": {
        1: "results_trimmed_mean_random_cifar_mean0_std01_j1", # mean 0 std 0.1
        2: "results_trimmed_mean_random_cifar_mean0_std01_j2",
        5: "results_trimmed_mean_random_cifar_mean0_std01_j5",
        10: "results_trimmed_mean_random_cifar_mean0_std01_j10",
    },
    "cifar10_2": {
        1: "results_trimmed_mean_random_cifar_mean1_std03_j1", # mean 1 std 0.3
        2: "results_trimmed_mean_random_cifar_mean1_std03_j2",
        5: "results_trimmed_mean_random_cifar_mean1_std03_j5",
        10: "results_trimmed_mean_random_cifar_mean1_std03_j10",
        # 20: "results_trimmed_mean_random_cifar_mean1_std03_j20",
    },
    "cifar10_3": {
        1: "results_trimmed_mean_random_cifar_mean1_j1", # mean 1 std 1
        2: "results_trimmed_mean_random_cifar_mean1_j2",
        5: "results_trimmed_mean_random_cifar_mean1_j5",
        10: "results_trimmed_mean_random_cifar_mean1_j10",
        # 20: "results_trimmed_mean_random_cifar_mean1_j20",
    },
}

DATASET_THRESHOLDS = {
    "mnist1": 85.0,
    "mnist2": 85.0,
    "mnist3": 85.0,
    "cifar10_1": 75.0,
    "cifar10_2": 75.0,
    "cifar10_3": 75.0,
}

GROUP_CATEGORY = {
    "mnist1": "mnist",
    "mnist2": "mnist",
    "mnist3": "mnist",
    "cifar10_1": "cifar",
    "cifar10_2": "cifar",
    "cifar10_3": "cifar",
}

DEFAULT_INIT_INFO = {
    "mnist1": {"mean": 0.0, "std": 1.0},
    "mnist2": {"mean": 3.0, "std": 1.0},
    "mnist3": {"mean": 5.0, "std": 2.0},
    "cifar10_1": {"mean": 0.0, "std": 0.1},
    "cifar10_2": {"mean": 10.0, "std": 0.3},
    "cifar10_3": {"mean": 1.0, "std": 1},
}


def _fmt_number(val):
    if val is None:
        return None
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    text = f"{val:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def infer_init_label(label: str, ji_map: dict) -> str | None:
    """Derive a readable init label (mean/std) from the j=1 config if available."""
    mean_val = None
    std_val = None

    j1_folder = ji_map.get(1)
    if j1_folder:
        config_path = os.path.join(j1_folder, "config.txt")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as cfg:
                for line in cfg:
                    key, _, raw_val = line.partition(":")
                    key = key.strip().lower()
                    raw_val = raw_val.strip()
                    if key == "mean":
                        try:
                            mean_val = float(raw_val)
                        except ValueError:
                            pass
                    elif key == "std":
                        try:
                            std_val = float(raw_val)
                        except ValueError:
                            pass

    defaults = DEFAULT_INIT_INFO.get(label, {})
    if mean_val is None:
        mean_val = defaults.get("mean")
    if std_val is None:
        std_val = defaults.get("std")

    mean_txt = _fmt_number(mean_val)
    std_txt = _fmt_number(std_val)

    if mean_txt and std_txt:
        return rf"$\mu={mean_txt},\ \sigma={std_txt}$"
    if mean_txt:
        return rf"$\mu={mean_txt}$"
    if std_txt:
        return rf"$\sigma={std_txt}$"
    return None


def compute_ratios(label: str, ji_map: dict, default_thr: float):
    series_data = {}
    max_values = {}

    # Preload all series for this label so we can choose a consistent threshold.
    for j, folder in ji_map.items():
        try:
            series = load_series(folder)
        except Exception as e:
            print(f"[{label}] load j={j} failed: {e}")
            continue
        series_data[j] = series
        max_values[j] = float(np.nanmax(series))

    if 1 not in series_data:
        print(f"[{label}] missing j=1 data, skip")
        return None

    j1_acc = series_data[1]
    max_j1 = max_values[1]
    if not np.isfinite(max_j1):
        print(f"[{label}] j1 max accuracy invalid, skip")
        return None

    actual_thr = min(default_thr, max_j1)
    if actual_thr < default_thr:
        print(f"[{label}] j1 never reaches {default_thr}%, fallback to max {actual_thr:.2f}%")

    # Ensure every available series can reach the threshold; otherwise lower it.
    available_max = [v for v in max_values.values() if np.isfinite(v)]
    if not available_max:
        print(f"[{label}] no valid series maxima, skip")
        return None
    min_max = min(available_max)
    if min_max < actual_thr:
        actual_thr = min_max
        print(f"[{label}] adjust threshold to {actual_thr:.2f}% so all j reach it")

    if actual_thr <= 0:
        print(f"[{label}] effective threshold <=0, skip")
        return None

    idx_j1 = first_reach_threshold(j1_acc, actual_thr)
    if idx_j1 <= 0 and actual_thr > 1e-6:
        idx_j1 = first_reach_threshold(j1_acc, actual_thr - 1e-6)
    if idx_j1 <= 0:
        print(f"[{label}] j1 didn't reach {actual_thr}% or step==0, skip")
        return None

    xs, speed_ratios, comm_ratios = [], [], []
    for j in sorted(series_data.keys()):
        acc = series_data[j]
        idx_ji = first_reach_threshold(acc, actual_thr)
        if idx_ji <= 0 and actual_thr > 1e-6:
            idx_ji = first_reach_threshold(acc, actual_thr - 1e-6)
        if idx_ji > 0:
            xs.append(j)
            speed_ratio = idx_j1 / idx_ji
            comm_ratio = (idx_j1 ) / (idx_ji * j)
            print(f"[{label}] j={j} speed_ratio={speed_ratio:.3f}, comm_ratio={comm_ratio:.3f}")
            speed_ratios.append(speed_ratio)
            comm_ratios.append(comm_ratio)
        else:
            print(f"[{label}] j={j} never reaches {actual_thr:.2f}%, skip in plot")

    if not xs or not speed_ratios or not comm_ratios:
        return None

    thr_label = f"{actual_thr:.0f}" if abs(actual_thr - round(actual_thr)) < 0.5 else f"{actual_thr:.1f}"
    return actual_thr, xs, speed_ratios, comm_ratios, thr_label


category_specs = [
    ("mnist", "MNIST"),
    ("cifar", "CIFAR-10"),
]

for cat, title in category_specs:
    outfile = f"ratio_vs_updates_{cat}.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plotted_speed = False
    plotted_comm = False
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    linestyles = ["-", "--", "-.", ":"]
    style_idx = 0

    for label, ji_map in mean_groups.items():
        if GROUP_CATEGORY.get(label) != cat:
            continue

        default_thr = DATASET_THRESHOLDS.get(label, 70.0)
        result = compute_ratios(label, ji_map, default_thr)
        if result is None:
            continue

        _, xs, speed_ratios, comm_ratios, _ = result 
        init_label = infer_init_label(label, ji_map)
        # 若能得到初始化信息，就用 LaTeX 标签；否则退回普通文本（可选：也包一层 $...$）
        label_text = init_label if init_label else label


        marker = markers[style_idx % len(markers)]
        linestyle = linestyles[style_idx % len(linestyles)]
        style_idx += 1

        axes[0].plot(
            xs,
            speed_ratios,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.2,
            markersize=7,
            label=label_text,
        )
        axes[1].plot(
            xs,
            comm_ratios,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.2,
            markersize=7,
            label=label_text,
        )
        plotted_speed = True
        plotted_comm = True

    label_kwargs = {"fontweight": "bold", "fontsize": 14}

    axes[0].set_ylabel("Communication Ratio", **label_kwargs)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].tick_params(axis="both", labelsize=12)
    if plotted_speed:
        axes[0].legend(
            title_fontproperties={"weight": "bold", "size": 12},
            prop={"weight": "bold", "size": 11},
        )
    else:
        axes[0].text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
            fontsize=12,
        )

    axes[1].set_ylabel("Computation Ratio", **label_kwargs)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].tick_params(axis="both", labelsize=12)
    if not plotted_comm:
        axes[1].text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=12,
        )

    fig.supxlabel("Multi Local Updates Parameter (J)", x=0.5, **label_kwargs)
    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    plt.close(fig)
