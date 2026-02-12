# gen_two_files.py
import argparse
import numpy as np
import torch

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def make_logistic_boundary_noise(
    n_samples=10000,
    n_features=2,
    w_star=None,
    b_star=None,
    x_dist="normal",        # "normal" | "uniform"
    feature_scale=1.0,
    seed=0,
    alpha=1.3,              # 距离衰减系数：越大=远离边界几乎不翻
    max_flip=0.30,          # 边界处最大翻转率（距离=0）
    clip_flip=0.49          # 安全上限（<0.5）
):
    rng = np.random.default_rng(seed)

    # 1) 生成/规范化 w*, b*
    if w_star is None:
        w_star = rng.normal(0, 1, size=n_features)
        w_star = w_star / (np.linalg.norm(w_star) + 1e-12) * 2.0
    if b_star is None:
        b_star = rng.normal(0, 0.3)

    # 2) 生成特征 X
    if x_dist == "normal":
        X = rng.normal(0, feature_scale, size=(n_samples, n_features))
    elif x_dist == "uniform":
        X = rng.uniform(-feature_scale, feature_scale, size=(n_samples, n_features))
    else:
        raise ValueError("x_dist must be 'normal' or 'uniform'.")

    # 3) 干净标签
    margin = X @ w_star + b_star
    p = sigmoid(margin)
    y_clean = rng.binomial(1, p).astype(np.int64)

    # 4) 距离相关翻转
    w_norm = np.linalg.norm(w_star) + 1e-12
    dist = np.abs(margin) / w_norm
    p_flip = np.minimum(max_flip * np.exp(-alpha * dist), clip_flip)

    flip_mask = rng.uniform(0, 1, size=n_samples) < p_flip
    y = y_clean.copy()
    y[flip_mask] = 1 - y[flip_mask]

    # 5) 用最优参数 quick-check（可不保存）
    acc_optimal = ((p >= 0.5).astype(np.int64) == y).mean()

    return X, y, w_star, float(b_star), float(acc_optimal)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--n_features", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--x_dist", type=str, default="normal", choices=["normal", "uniform"])
    ap.add_argument("--feature_scale", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.4)
    ap.add_argument("--max_flip", type=float, default=0.25)
    ap.add_argument("--clip_flip", type=float, default=0.49)
    ap.add_argument("--out_dataset", type=str, default="dataset.pt")
    ap.add_argument("--out_wstar", type=str, default="wstar.pt")
    args = ap.parse_args()

    X, y, w_star, b_star, acc_opt = make_logistic_boundary_noise(
        n_samples=args.n_samples,
        n_features=args.n_features,
        seed=args.seed,
        x_dist=args.x_dist,
        feature_scale=args.feature_scale,
        alpha=args.alpha,
        max_flip=args.max_flip,
        clip_flip=args.clip_flip
    )

    # 保存成 torch 能直接用的格式
    X_t = torch.from_numpy(X).to(torch.float32)
    y_t = torch.from_numpy(y).to(torch.long)
    torch.save({"X": X_t, "y": y_t}, args.out_dataset)

    w_t = torch.from_numpy(w_star).to(torch.float32)
    b_t = torch.tensor(b_star, dtype=torch.float32)
    torch.save({"w_star": w_t, "b_star": b_t}, args.out_wstar)

    print(f"[Saved] dataset -> {args.out_dataset}  (X: {tuple(X_t.shape)}, y: {tuple(y_t.shape)})")
    print(f"[Saved] w*      -> {args.out_wstar}   (||w*||={w_t.norm().item():.3f}, b*={b_t.item():.3f})")
    print(f"acc@optimal (on noisy labels) ≈ {acc_opt:.4f}")

if __name__ == "__main__":
    main()
