# eval_optimal.py
import torch
import argparse

def eval_with_wstar(dataset_pt: str, wstar_pt: str):
    ds = torch.load(dataset_pt, map_location="cpu")
    ws = torch.load(wstar_pt, map_location="cpu")
    X, y = ds["X"], ds["y"]
    w, b = ws["w_star"], ws["b_star"]

    logits = X @ w + b
    pred = (torch.sigmoid(logits) >= 0.5).to(torch.long)
    acc = (pred == y).float().mean().item()
    return acc, X.shape[0], X.shape[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="dataset.pt")
    ap.add_argument("--wstar", type=str, default="wstar.pt")
    args = ap.parse_args()

    acc, n, d = eval_with_wstar(args.dataset, args.wstar)
    print(f"[Eval] n={n} d={d} acc@optimal={acc:.4f}")
