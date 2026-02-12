# securedl.py

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


# ============================== 工具：参数向量/列表互转 ==============================

def _params_to_vec(model: nn.Module, device: torch.device) -> torch.Tensor:
    parts = [p.detach().to(device).view(-1) for p in model.parameters()]
    return torch.cat(parts) if parts else torch.tensor([], device=device)

@torch.no_grad()
def _vec_to_params_(model: nn.Module, vec: torch.Tensor):
    off = 0
    for p in model.parameters():
        n = p.numel()
        if n > 0:
            p.copy_(vec[off:off+n].view_as(p))
        off += n

def _list_to_vec(params_list: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    parts = [p.to(device).view(-1) for p in params_list]
    return torch.cat(parts) if parts else torch.tensor([], device=device)

def _vec_to_list(vec: torch.Tensor, like: List[torch.Tensor]) -> List[torch.Tensor]:
    out, off = [], 0
    for p in like:
        n = p.numel()
        out.append(vec[off:off+n].view_as(p).detach().cpu().clone())
        off += n
    return out


# ============================== 本地训练（作为 half-step 消息） ==============================

def _train_one_node(
    node_idx: int,
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    current_lr: float,
    local_epochs: int,
):
    """
    返回: node_idx, avg_loss, trained_params(List[Tensor], 已搬到 CPU)
    - 这里输出的是节点本轮“本地更新后”的参数（作为要广播的 half-step）
      为保持与你现有代码兼容：纯 SGD（无动量），local_epochs 次遍历 dataloader。
    - 如需换成动量，也可在不改接口的前提下替换为动量 half-step。
    """
    model_copy = copy.deepcopy(model).to(device)
    model_copy.train()

    last_losses = []
    for _ in range(local_epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            out  = model_copy(data)
            loss = criterion(out, target)

            model_copy.zero_grad(set_to_none=True)
            loss.backward()

            # SGD 更新
            with torch.no_grad():
                for p in model_copy.parameters():
                    if p.grad is not None:
                        p.add_( -current_lr * p.grad )

        last_losses.append(float(loss.item()))

    avg_loss = float(np.mean(last_losses)) if last_losses else float("nan")
    trained_params = [p.data.detach().cpu().clone() for p in model_copy.parameters()]
    return node_idx, avg_loss, trained_params


# ============================== SecureDL：单 epoch ==============================

def secure_epoch(
    models: List[nn.Module],
    trainloaders: List[torch.utils.data.DataLoader],
    adj_matrix: np.ndarray,                 # 0/1 或 bool，对称拓扑（无向）
    byzantine_indices: List[int],
    criterion: nn.Module,
    current_lr: float,
    defense_obj: Optional[BaseDefense],     # 保留接口，不使用
    attack_type: str,
    config: Config,
    epoch: int,
):
    """
    SecureDL 聚合（接口与返回与 rtc_epoch 一致）：
      1) 本地训练得到要广播的参数 x_j^{half}
      2) （可选）攻击：替换拜占庭节点的广播
      3) 对每个节点 i：
         - 定义更新向量（“half vector”对应的 delta）：
             u_i = x_i^{half} - x_i^t
             u_j = x_j^{half} - x_j^t       # 注意：各自相对各自快照
         - 计算 cos(u_i, u_j)，若 cos < τ 则丢弃 j（可选关闭）
         - 对通过的邻居 j，做 L2 归一（或可选关闭）：
             u'_j = u_j * (||u_i|| / ||u_j||)   # 保留 j 的方向，仅拉齐范数到 i
         - 聚合：u_hat = mean( {u_i} ∪ {u'_j, j通过} ∪ {必要时从拒绝集中补齐Top-K} )
         - 全局步长：x_i^{t+1} = x_i^t + α * u_hat
      4) 回写 x^{t+1}_i 到模型

    需要的超参（从 config 读取，若无则用默认）：
      - securedl_tau           ∈ [0,1]：余弦阈值 τ（默认 0.0，即不过滤）
      - securedl_alpha         ∈ (0, +∞)：全局聚 合步长 α（默认 0.5）
      - securedl_min_keep      ∈ N0：每个节点至少保留的邻居数（默认 2）
      - securedl_disable_norm  : bool  关闭 L2 归一化（默认 False）
      - securedl_disable_cosine: bool  关闭余弦筛选（默认 False）
    """
    num_nodes = config.num_nodes
    device    = config.device
    eps       = 1e-12

    # 超参
    tau      = float(getattr(config, "securedl_tau",   0.3))     # 余弦阈值
    g_alpha  = float(getattr(config, "securedl_alpha", 1))     # 聚合步长
    min_keep = int(getattr(config, "securedl_min_keep", 2))      # 至少保留 K 个邻居
    disable_norm   = bool(getattr(config, "securedl_disable_norm", True))
    disable_cosine = bool(getattr(config, "securedl_disable_cosine", True))

    # -------- x^t 快照（圆心）--------
    x_t_snapshots: List[List[torch.Tensor]] = [[p.detach().cpu().clone() for p in m.parameters()] for m in models]
    x_t_vecs = [ _list_to_vec(x_t_snapshots[i], device) for i in range(num_nodes) ]

    # -------- 本地训练，得到要广播的 half-step 参数 --------
    trained_params = [None] * num_nodes
    losses         = [float("nan")] * num_nodes
    for i in range(num_nodes):
        n, l, tparams = _train_one_node(
            i, models[i], trainloaders[i], device,
            criterion, current_lr, config.local_epochs,
        )
        losses[n]         = l
        trained_params[n] = tparams  # list[Tensor] (CPU)

    # -------- （可选）拜占庭攻击：替换要广播的数据 --------
    params_to_send: List[List[torch.Tensor]] = [[p.clone() for p in trained_params[i]] for i in range(num_nodes)]
    if attack_type and attack_type.lower() != "none":
        honest_idx   = [i for i in range(num_nodes) if i not in byzantine_indices]
        honest_param = [[p.to(device) for p in trained_params[i]] for i in honest_idx]
        for byz in byzantine_indices:
            fake = get_byzantine_params(
                [p.to(device) for p in trained_params[byz]],   # 基于 half-step
                attack_type,
                honest_params=honest_param,
                config=config,
                device=device,
            )
            params_to_send[byz] = [p.detach().cpu().clone() for p in fake]
            # 写回本地模型（影响下一轮起点）
            with torch.no_grad():
                for pm, pf in zip(models[byz].parameters(), fake):
                    pm.data.copy_(pf)

    # -------- SecureDL 聚合 --------
    # 预先把要广播的参数转 vec（GPU）
    x_half_vecs = [ _list_to_vec(params_to_send[j], device) for j in range(num_nodes) ]

    # 结果容器（vec）
    x_next_vecs: List[torch.Tensor] = [ None for _ in range(num_nodes) ]  # type: ignore

    for i in range(num_nodes):
        # 邻居集合（含 self）
        neighbors_i = [j for j in range(num_nodes) if (j == i or adj_matrix[i, j])]

        x_i_t    = x_t_vecs[i]       # 圆心（上轮参数）
        x_i_half = x_half_vecs[i]    # 本地 half-step

        # 自身更新向量
        u_i = (x_i_half - x_i_t)
        norm_ui = torch.norm(u_i, p=2).clamp_min(eps)

        # 候选集合：始终包含自己
        accepted_updates = [u_i]

        # 用于“至少保留 K 个”的池子
        # 分成两类：通过阈值的 accepted_pool 与未过阈值的 rejected_pool
        accepted_pool = []  # (cos_val, j, u_j_scaled)
        rejected_pool = []  # (cos_val, j, u_j_scaled)

        # 邻居筛选（不含 self）
        for j in neighbors_i:
            if j == i:
                continue

            x_j_t = x_t_vecs[j]
            u_j   = (x_half_vecs[j] - x_j_t)
            norm_uj = torch.norm(u_j, p=2).clamp_min(eps)

            # 计算相似度；允许关闭筛选
            if disable_cosine:
                cos_val = 1.0
            else:
                cos_ij  = torch.dot(u_i, u_j) / (norm_ui * norm_uj)
                cos_val = float(cos_ij.clamp(-1.0, 1.0).item())

            # 归一化；允许关闭
            if disable_norm:
                u_j_scaled = u_j
            else:
                # L2 归一：保留 j 的方向，把范数拉到与 u_i 相同
                u_j_scaled = u_j * (norm_ui / norm_uj)

            # 根据阈值分拣
            if cos_val >= tau:
                accepted_pool.append((cos_val, j, u_j_scaled))
            else:
                rejected_pool.append((cos_val, j, u_j_scaled))

        # 先全部加入“已通过阈值”的邻居
        if len(accepted_pool) > 0:
            accepted_pool.sort(key=lambda x: x[0], reverse=True)
            for _, _, uj in accepted_pool:
                accepted_updates.append(uj)

        # 若通过人数不足 K，则从拒绝池中按相似度补足
        if min_keep > 0 and (len(accepted_updates) - 1) < min_keep:  # -1 去掉 self
            need = min_keep - (len(accepted_updates) - 1)
            if len(rejected_pool) > 0 and need > 0:
                rejected_pool.sort(key=lambda x: x[0], reverse=True)
                for _, _, uj in rejected_pool[:need]:
                    accepted_updates.append(uj)

        # 均值聚合 + 全局步长
        # 若无任何邻居通过（只有自身），也能正常运行：mean 即 u_i
        u_hat  = torch.mean(torch.stack(accepted_updates, dim=0), dim=0)
        x_next = x_i_t + g_alpha * u_hat
        x_next_vecs[i] = x_next.detach()

    # -------- 回写 x^{t+1} 到模型参数 --------
    for i in range(num_nodes):
        _vec_to_params_(models[i], x_next_vecs[i].to(device))

    global_loss = float(np.nanmean(losses))
    return global_loss, losses
