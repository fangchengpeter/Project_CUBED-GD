# rtc_strict_full.py
# 严格按论文实现：Algorithm 2（动量 + half-step 通信） + Algorithm 1（Remove-then-Clip）
# 同时提供 L 的估计与 α := 4ηL 的设定工具；保持你原有函数签名与返回值不变。

import copy
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

from config import Config
try:
    from defenses import BaseDefense
except ImportError:
    BaseDefense = object

from byzantine import get_byzantine_params


# ============================== 工具：参数向量化/反向量化 ==============================

def _params_to_vec(model: nn.Module, device: torch.device) -> torch.Tensor:
    parts = [p.detach().to(device).view(-1) for p in model.parameters()]
    return torch.cat(parts) if parts else torch.tensor([], device=device)

@torch.no_grad()
def _vec_to_params_(model: nn.Module, vec: torch.Tensor):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        if n > 0:
            p.copy_(vec[offset:offset+n].view_as(p))
        offset += n

def _list_to_vec(params_list: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    parts = [p.to(device).view(-1) for p in params_list]
    return torch.cat(parts) if parts else torch.tensor([], device=device)

def _vec_to_list(vec: torch.Tensor, like: List[torch.Tensor]) -> List[torch.Tensor]:
    out, offset = [], 0
    for p in like:
        n = p.numel()
        out.append(vec[offset:offset+n].view_as(p).detach().cpu().clone())
        offset += n
    return out


# ============================== 工具：估计 L 与设置 α := 4ηL ==============================

def estimate_spectral_norm_sq(trainloader: torch.utils.data.DataLoader,
                              device: torch.device,
                              iters: int = 5) -> float:
    """
    近似 ||A||_2^2：对每个 batch（展平为 B×d）做 power iteration，取最大值。
    适用逻辑回归的 L 上界：L <= 0.25*||A||_2^2 + θ
    """
    s_max_sq = 0.0
    for xb, _ in trainloader:
        X = xb.to(device).float().view(xb.size(0), -1)
        if X.numel() == 0:
            continue
        v = torch.randn(X.size(1), 1, device=device)
        v = v / (v.norm() + 1e-12)
        for _ in range(iters):
            u = X @ v
            v = X.t() @ u
            v = v / (v.norm() + 1e-12)
        u = X @ v
        sigma = (u.norm() + 1e-12)
        s_max_sq = max(s_max_sq, float(sigma.item() ** 2))
    return s_max_sq

def set_eta_alpha_strict(models: List[nn.Module],
                         trainloaders: List[torch.utils.data.DataLoader],
                         device: torch.device,
                         theta_l2: float,
                         mode: str = "A",
                         target_alpha: float = 0.4,
                         eta_fixed: Optional[float] = None) -> Tuple[float, float, float]:
    """
    设定 (η, α) 使 α := 4ηL。
    mode="A": 固定 α=target_alpha，反解 η=α/(4L)（推荐，更稳）
    mode="B": 固定 η=eta_fixed，设 α=clip(4ηL, (1e-3,0.9))
    返回: (eta, alpha, L_hat)
    """
    L_list = []
    for ld in trainloaders:
        s2 = estimate_spectral_norm_sq(ld, device, iters=5)
        L_i = 0.25 * s2 + theta_l2
        L_list.append(L_i)
    L_hat = max(L_list) if L_list else 1.0

    if mode == "A":
        alpha = target_alpha
        eta   = alpha / (4.0 * L_hat + 1e-12)
    else:
        assert eta_fixed is not None, "mode='B' 需要提供 eta_fixed"
        eta   = eta_fixed
        alpha = max(1e-3, min(0.9, 4.0 * eta * L_hat))

    for m in models:
        m._rtc_alpha = float(alpha)

    return float(eta), float(alpha), float(L_hat)


# ============================== 本地计算：动量 + half-step（Algorithm 2 Lines 4–6） ==============================

def _train_one_node(
    node_idx: int,
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    current_lr: float,          # 论文中的 η
    local_epochs: int,          # 建议=1 以贴近算法
):
    """
    返回: node_idx, avg_loss, trained_params(List[Tensor], 已搬到 CPU)
    - 严格按论文：返回 x^{t+1/2}（half-step）
    - 动量 m_{t+1} = (1-α)m_t + α*g_t 保存在 model._rtc_momentum 中
    - α 从 model._rtc_alpha 读取（由 set_eta_alpha_strict 预先设置）
    """
    # x^t
    x_vec = _params_to_vec(model, device=device)

    # 动量
    if not hasattr(model, "_rtc_momentum"):
        model._rtc_momentum = torch.zeros_like(x_vec, device=device)
    m_vec = model._rtc_momentum
    alpha = getattr(model, "_rtc_alpha", getattr(model, "alpha", 0.3))

    model.train()
    last_losses = []
    it = iter(trainloader)

    for _ in range(local_epochs):
        try:
            data, target = next(it)
        except StopIteration:
            it = iter(trainloader)
            data, target = next(it)
        data, target = data.to(device), target.to(device)

        model.zero_grad(set_to_none=True)
        out  = model(data)
        loss = criterion(out, target)
        loss.backward()

        # g_t 向量
        g_parts = []
        for p in model.parameters():
            if p.grad is None:
                g_parts.append(torch.zeros_like(p, device=device).view(-1))
            else:
                g_parts.append(p.grad.detach().to(device).view(-1))
        g_vec = torch.cat(g_parts) if g_parts else torch.tensor([], device=device)

        # m_{t+1} 与 half-step
        m_vec = (1.0 - alpha) * m_vec + alpha * g_vec
        x_vec = x_vec - current_lr * m_vec

        last_losses.append(float(loss.item()))
        _vec_to_params_(model, x_vec)

    model._rtc_momentum = m_vec.detach().clone()

    trained_params = _vec_to_list(x_vec, [p.detach() for p in model.parameters()])
    avg_loss = float(np.mean(last_losses)) if last_losses else float("nan")
    return node_idx, avg_loss, trained_params


# ============================== 单 epoch：half-step + 攻击 + RTC 聚合（Algorithm 1 & 2） ==============================

def rtc_epoch(
    models: List[nn.Module],
    trainloaders: List[torch.utils.data.DataLoader],
    adj_matrix: np.ndarray,                 # 0/1 或 bool，对称拓扑
    byzantine_indices: List[int],
    criterion: nn.Module,
    current_lr: float,
    defense_obj: Optional[BaseDefense],     # 未用；占位保持接口
    attack_type: str,
    config: Config,
    epoch: int,
):
    """
    - 圆心使用 x^t（本轮开始前的快照）
    - 本地计算得到 x^{t+1/2} 并作为通信消息（若有攻击则改这个消息）
    - RTC：先移除（累计权重 ≤ δ_max,i），再以 x^t_i 为圆心裁剪，半径 τ_i 按式(10)估计
    - W 按式(15)构造为对称双随机
    返回: global_loss, losses(list)
    """
    num_nodes = config.num_nodes
    device    = config.device
    eps       = 1e-12

    # x^t 快照（圆心）
    snapshots = [[p.detach().cpu().clone() for p in m.parameters()] for m in models]

    # 本地 half-step
    trained_params = [None] * num_nodes   # 作为 x^{t+1/2}
    losses         = [float("nan")] * num_nodes
    for i in range(num_nodes):
        
        n, l, tparams = _train_one_node(
            i, models[i], trainloaders[i], device,
            criterion, current_lr, config.local_epochs,
        )
        losses[n]         = l
        trained_params[n] = tparams

    # 广播 & 攻击（对 half-step）
    params_to_send: List[List[torch.Tensor]] = [[p.clone() for p in trained_params[i]] for i in range(num_nodes)]
    if attack_type and attack_type.lower() != "none":
        honest_idx   = [i for i in range(num_nodes) if i not in byzantine_indices]
        honest_param = [[p.to(device) for p in trained_params[i]] for i in honest_idx]
        for byz in byzantine_indices:
            fake = get_byzantine_params(
                [p.to(device) for p in trained_params[byz]],
                attack_type,
                honest_params=honest_param,
                config=config,
                device=device,
            )
            params_to_send[byz] = [p.detach().cpu().clone() for p in fake]
            with torch.no_grad():
                for pm, pf in zip(models[byz].parameters(), fake):
                    pm.data.copy_(pf)

    # 构造 W（式 15）
    degs  = adj_matrix.sum(axis=1).astype(int)
    dmax  = int(degs.max()) if num_nodes > 0 else 0
    base  = 1.0 / float(dmax + 1)
    W = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        W[i, i] = 1.0 - degs[i] * base
        for j in range(num_nodes):
            if i != j and adj_matrix[i, j]:
                W[i, j] = base

    byz_set = set(byzantine_indices)

    # 工具
    def _l2(xa: List[torch.Tensor], xb: List[torch.Tensor]) -> float:
        s = 0.0
        for pa, pb in zip(xa, xb):
            d = (pa - pb).view(-1)
            s += float(torch.dot(d, d).item())
        return s ** 0.5

    def _axpy(out_list: List[torch.Tensor], a: float, x_list: List[torch.Tensor]):
        for i_, x in enumerate(x_list):
            out_list[i_].add_(x, alpha=a)

    # RTC 聚合
    aggregated_params_cpu: List[List[torch.Tensor]] = [None] * num_nodes

    for i in range(num_nodes):
        neighbors_i = [j for j in range(num_nodes) if (j == i or adj_matrix[i, j])]

        x_center_i = snapshots[i]          # x^t_i（圆心）
        x_half_i   = params_to_send[i]     # x^{t+1/2}_i

        # δ_max,i = 邻域内（不含 self）的 Byzantine 权重和（严格按论文）
        delta_max_i = 0.0
        for j in neighbors_i:
            if j == i:
                continue
            if j in byz_set:  # byzantine_indices 提供的真实集合
                delta_max_i += float(W[i, j])


        # 候选（不含 self），按距离降序
        triples = []
        for j in neighbors_i:
            if j == i:
                continue
            x_half_j = params_to_send[j]
            dist = _l2(x_center_i, x_half_j)   # 以 x^t_i 为圆心
            triples.append((j, dist, float(W[i, j]), x_half_j))
        triples.sort(key=lambda t: t[1], reverse=True)

        # Remove：累计权重 ≤ δ_max,i
        removed, removed_w = set(), 0.0
        for j, dist, w, _ in triples:
            if removed_w + w <= delta_max_i + eps:
                removed.add(j)
                removed_w += w
            else:
                continue

        # 剩余 S_i（强制含 self）
        remain = [(j, dist, w, xj) for (j, dist, w, xj) in triples if j not in removed and w > 0.0]
        remain.append((i, 0.0, float(W[i, i]), x_half_i))

        # 退化：δ=0 => 加权平均 x^{t+1/2}
        if delta_max_i <= 0.0 or len(remain) == 0:
            out = [torch.zeros_like(p) for p in x_center_i]
            for j, _dist, w, xj in remain:
                _axpy(out, w, xj)
            aggregated_params_cpu[i] = out
            continue

        # τ_i = sqrt( (1/δ_max,i) * sum_{j∈S_i} w_ij * ||x^t_i - x^{t+1/2}_j||^2 )
        num_tau = sum(w * (dist ** 2) for (_j, dist, w, _xj) in remain)
        tau_sq = (num_tau / max(delta_max_i, eps))

        # 聚合：移除者 -> w*x^t_i；剩余者 -> w*(x^t_i + CLIP(x^{t+1/2}_j - x^t_i, τ_i))
        out = [torch.zeros_like(p) for p in x_center_i]

        for j, dist, w, _ in triples:
            if j in removed and w > 0.0:
                _axpy(out, w, x_center_i)

        for j, dist, w, xj in remain:
            if w <= 0.0:
                continue
            if dist * dist <= tau_sq + 1e-12:
                scale = 1.0
            else:
                scale = tau_sq / (dist * dist + 1e-12)

            clipped = [pi + (pj - pi) * scale for pi, pj in zip(x_center_i, xj)]
            _axpy(out, w, clipped)

        i0 = 0
        if i == i0:
            kept = [(j, dist, w) for (j, dist, w, _) in remain]
            removed_ids = [j for (j, _, _, _) in triples if j in removed]
            kept_dists  = [dist for (_, dist, _) in kept]
            print(f"[epoch {epoch}] node {i0}: "
                f"delta_max={delta_max_i:.3f}, removed_w={removed_w:.3f}, "
                f"removed={removed_ids}, tau_sq={tau_sq:.3e}, "
                f"kept_mean||u||={np.mean(kept_dists):.3e}, P90={np.quantile(kept_dists,0.9):.3e}")


        aggregated_params_cpu[i] = out

    # 回写到模型
    for i in range(num_nodes):
        with torch.no_grad():
            for pm, pnew in zip(models[i].parameters(), aggregated_params_cpu[i]):
                pm.data.copy_(pnew.to(device))

    global_loss = float(np.nanmean(losses))
    return global_loss, losses
