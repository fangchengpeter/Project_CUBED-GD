import copy
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from config import Config               
try:
    from defenses import BaseDefense
except ImportError:
    BaseDefense = object

from byzantine import get_byzantine_params   


# ------------------------------------------------------------
# 单节点本地训练：接口保持不变
# ------------------------------------------------------------
def _train_one_node(
    node_idx: int,
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    current_lr: float,
    local_epochs: int,
    penalty: List[torch.Tensor],           # Σ sign(x_i - x_j)，与每层 param 对齐 (CPU or GPU)
):
    """
    返回: node_idx, avg_loss, trained_params(List[Tensor], 已搬到 CPU)
    """
    # -------- 读取全局正则系数 --------
    tv_lambda = 0.005

    model_copy = copy.deepcopy(model).to(device)
    model_copy.train()

    last_losses = []

    for _ in range(local_epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            out   = model_copy(data)
            loss  = criterion(out, target)


            # ==== 反向传播 ====
            model_copy.zero_grad()
            loss.backward()

            # ==== 加 TV 梯度项 (λ * Σ sign) ====
            if tv_lambda > 0.0:
                for p, s in zip(model_copy.parameters(), penalty):
                    if p.grad is not None:
                        p.grad += tv_lambda * s.to(device)

            # ==== SGD 更新 ====
            with torch.no_grad():
                for p in model_copy.parameters():
                    if p.grad is not None:
                        p.data -= current_lr * p.grad

        last_losses.append(loss.item())

    avg_loss = float(np.mean(last_losses)) if last_losses else float('nan')
    trained_params = [p.data.clone().cpu() for p in model_copy.parameters()]
    return node_idx, avg_loss, trained_params


# ------------------------------------------------------------
# 单 epoch：聚合-训练-攻击
# ------------------------------------------------------------
def braso_epoch(
    models: List[nn.Module],
    trainloaders: List[torch.utils.data.DataLoader],
    adj_matrix: np.ndarray,
    byzantine_indices: List[int],
    criterion: nn.Module,
    current_lr: float,
    defense_obj: Optional[BaseDefense],   # 不再使用，仅为兼容签名
    attack_type: str,
    config: Config,
    epoch: int,
):
    """
      1) 每个节点广播本轮的模型快照（拜占庭节点发 z_j^k）
      2) 每个正则节点计算 TV sign 和 (含拜占庭项)
      3) 每个正则节点做一次 SGD 更新：x_i^{k+1} = x_i^k - α_k(∇F + λΣsign + ∇f0)
    注：这里把 ∇f0 通过把 penalty 直接加到 grad 的方式等效实现；若你有显式 f0，请在 loss 中加入。
    """
    num_nodes  = config.num_nodes
    device     = config.device
    tv_lambda  = getattr(config, "tv_lambda", 0.005)   # 论文里的 λ，默认 0.005，可在 Config 里改
    local_epochs = 1                                   # 论文严格是“每步一次”随机次梯度

    # ---------- (k) 时刻的模型快照：作为本轮的广播基准 ----------
    snapshots = [[p.detach().cpu().clone() for p in m.parameters()] for m in models]

    # ---------- 生成拜占庭广播 z_j^k（对所有邻居统一 z_j^k；如需 per-neighbor 可在此细化） ----------
    broadcast_params = [None] * num_nodes
    if attack_type and attack_type.lower() != "none" and len(byzantine_indices) > 0:
        honest_idx   = [i for i in range(num_nodes) if i not in byzantine_indices]
        honest_param = [[p.to(device) for p in snapshots[i]] for i in honest_idx]

        for j in range(num_nodes):
            if j in byzantine_indices:
                fake = get_byzantine_params(
                    [p.to(device) for p in snapshots[j]],
                    attack_type,
                    honest_params=honest_param,
                    config=config,
                    device=device,
                )
                broadcast_params[j] = [p.detach().cpu().clone() for p in fake]
            else:
                broadcast_params[j] = snapshots[j]
    else:
        # 无攻击：广播真实模型
        broadcast_params = snapshots

    # ---------- 为每个节点 i 计算 Σ sign(x_i^k - m_j^k)，m_j^k 为本轮邻居广播 ----------
    penalty_list: List[List[torch.Tensor]] = []
    for i in range(num_nodes):
        neigh_idxs = [j for j, connect in enumerate(adj_matrix[i]) if connect]  # 包含自身或仅邻居均可；论文里用邻居
        s = [torch.zeros_like(p) for p in snapshots[i]]  # 和每层 param 对齐（CPU 张量）
        for j in neigh_idxs:
            for k, (p_i, m_j) in enumerate(zip(snapshots[i], broadcast_params[j])):
                s[k] += torch.sign(p_i - m_j)  # element-wise sign
        penalty_list.append(s)

    # ---------- 单步本地更新（严格“每轮一次”随机次梯度；不做任何聚合） ----------
    trained_params = [None] * num_nodes
    losses         = [float('nan')] * num_nodes

    for i in range(num_nodes):
        # 取一个 batch（若 DataLoader 是无限迭代器更好；否则这里 next 用 try/except）
        trainloader = trainloaders[i]
        try:
            data, target = next(iter(trainloader))
        except TypeError:
            # 某些 DataLoader 不支持直接 iter 重用，退回 for-break 取一个 batch
            it = iter(trainloader)
            data, target = next(it)
        data, target = data.to(device), target.to(device)

        model = models[i]
        model.train()

        model.zero_grad()
        out  = model(data)
        loss = criterion(out, target)

        # 反向传播
        loss.backward()

        # 加 TV sign 梯度项（λ * Σ sign），按论文式 (5)
        if tv_lambda > 0.0:
            for p, s in zip(model.parameters(), penalty_list[i]):
                if p.grad is not None:
                    p.grad += tv_lambda * s.to(device)

        # SGD 一步
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.data -= current_lr * p.grad

        # 记录
        losses[i] = float(loss.item())
        trained_params[i] = [p.detach().cpu().clone() for p in model.parameters()]

    # 没有任何“聚合/防御覆盖”；每个节点已经得到 x_i^{k+1} 留待下一轮广播
    global_loss = float(np.nanmean(losses))
    return global_loss, losses
