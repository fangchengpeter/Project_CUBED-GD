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
    "mnist": {
        1: "results_trimmed_mean_mnist_j1_baseline",
        2: "results_trimmed_mean_mnist_j2_baseline",
        5: "results_trimmed_mean_mnist_j5_baseline",
        10: "results_trimmed_mean_mnist_j10_baseline"
    },
    "cifar10": {
        1: "results_j1_baseline",
        2: "results_j2_baseline",
        5: "results_j5_baseline",
        10: "results_j10_baseline",
    }
}

DATASET_THRESHOLDS = {
    "mnist": 90.0,
    "cifar10": 80.0,
}

plt.figure(figsize=(8, 5))

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]

for idx, (label, ji_map) in enumerate(mean_groups.items()):
    marker = MARKERS[idx % len(MARKERS)]
    linestyle = LINESTYLES[idx % len(LINESTYLES)]

    thr = DATASET_THRESHOLDS.get(label, 70.0)

    j1_path = ji_map.get(1)
    if j1_path is None:
        continue

    try:
        j1_acc = load_series(j1_path)
    except Exception as e:
        print(f"[{label}] load j1 failed: {e}")
        continue

    idx_j1 = first_reach_threshold(j1_acc, thr)
    if idx_j1 <= 0:
        print(f"[{label}] j1 didn't reach {thr}% or step==0, skip")
        continue

    xs, ys = [], []
    for j, folder in sorted(ji_map.items(), key=lambda x: x[0]):
        try:
            acc = load_series(folder)
        except Exception as e:
            print(f"[{label}] load j={j} failed: {e}")
            continue
        idx_ji = first_reach_threshold(acc, thr)
        if idx_ji > 0:
            ratio = idx_j1 / idx_ji
            print(f"[{label}] j={j} ratio={ratio:.3f}")
            xs.append(j)
            ys.append(ratio)

    if xs and ys:
        plt.plot(
            xs,
            ys,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=7,
            label=f"{label} (thr={thr:.0f}%)",
        )

plt.xlabel("j")
plt.ylabel("steps_to_threshold ratio (Nc(~)/Nc)")
plt.title("Time-to-threshold accuracy ratio (MNIST thr=90, CIFAR-10 thr=80)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(title="dataset")
plt.tight_layout()
plt.savefig("distance_vs_j_ratio.png", dpi=160)
plt.show()
