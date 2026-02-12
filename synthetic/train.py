# train_lib.py
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class LogisticRegression(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)  # 输出 logits
    def forward(self, x):
        return self.linear(x).squeeze(-1)

def _accuracy_from_logits(logits, y):
    pred = (logits >= 0).to(torch.long)
    return (pred == y).float().mean().item()

def train_logreg(
    dataset_path: str,
    *,
    lr: float = 1e-2,
    epochs: int = 50,
    batch_size: int = 1024,
    weight_decay: float = 0.0,
    device: str = "cpu",
):
    """
    读取 dataset.pt（含 X: float32, y: int64）。
    使用纯手写 SGD（无 optimizer）训练逻辑回归并返回 (model, metrics)。

    metrics = {
      'history': [{'epoch': i, 'loss': ..., 'acc': ...}, ...],
      'final': {'acc_all': ..., 'n': N, 'd': D}
    }
    """
    # ---- 数据 ----
    bundle = torch.load(dataset_path, map_location="cpu")
    X, y = bundle["X"].to(device), bundle["y"].to(device)
    n, d = X.shape
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True, drop_last=False)

    # ---- 模型 & 损失 ----
    model = LogisticRegression(d).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_count = 0.0, 0, 0

        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb.float())

            # 清梯度
            for p in model.parameters():
                p.grad = None
            # 反传
            loss.backward()

            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        if weight_decay != 0.0:
                            p.grad.add_(weight_decay * p)   # L2 正则
                        p.add_(-lr * p.grad)

            total_loss += loss.item() * xb.size(0)
            total_correct += (logits >= 0).to(torch.long).eq(yb).sum().item()
            total_count += xb.size(0)

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count
        log = {"epoch": epoch, "loss": train_loss, "acc": train_acc}
        history.append(log)

    # ---- 全量评估 ----
    with torch.no_grad():
        logits_all = model(X)
        acc_all = _accuracy_from_logits(logits_all, y)
        final = {"acc_all": acc_all, "n": int(n), "d": int(d)}

    metrics = {"history": history, "final": final}
    return model, metrics

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="dataset.pt")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=10000)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_model", type=str, default="model.pt")
    ap.add_argument("--out_metrics", type=str, default="metrics.json")
    args = ap.parse_args()

    model, metrics = train_logreg(
        args.dataset,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        device=args.device,
    )

    torch.save(model.state_dict(), args.out_model)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Saved] model   -> {args.out_model}")
    print(f"[Saved] metrics -> {args.out_metrics}")
    print(f"Final accuracy: {metrics['final']['acc_all']:.4f} (n={metrics['final']['n']}, d={metrics['final']['d']})")