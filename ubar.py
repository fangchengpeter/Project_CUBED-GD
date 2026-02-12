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


def _train_one_node(
    node_idx: int,
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    current_lr: float,
    local_epochs: int,
    penalty: List[torch.Tensor],  # 为了兼容接口保留；不使用
):
    """
    返回: node_idx, avg_loss, trained_params(List[Tensor], 已搬到 CPU)
    """
    model_copy = copy.deepcopy(model).to(device)
    model_copy.train()

    last_losses = []

    for _ in range(local_epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)

            out  = model_copy(data)
            loss = criterion(out, target)

            model_copy.zero_grad()
            loss.backward()

            # 纯 SGD 更新
            with torch.no_grad():
                for p in model_copy.parameters():
                    if p.grad is not None:
                        p.data -= current_lr * p.grad

        last_losses.append(loss.item())

    avg_loss = float(np.mean(last_losses)) if last_losses else float('nan')
    trained_params = [p.data.clone().cpu() for p in model_copy.parameters()]
    return node_idx, avg_loss, trained_params


# ------------------------------------------------------------
# 单 epoch：本地训练 +（可选）攻击 + UBAR 两阶段聚合 + α 混合
# ------------------------------------------------------------
def ubar_epoch(
    models: List[nn.Module],
    trainloaders: List[torch.utils.data.DataLoader],
    adj_matrix: np.ndarray,                 # 0/1 或 bool，对称拓扑
    byzantine_indices: List[int],
    criterion: nn.Module,
    current_lr: float,
    defense_obj: Optional[BaseDefense],
    attack_type: str,
    config: Config,
    epoch: int,
):
    """
    与论文一致的 UBAR 实现：
      - 阶段1：按与自身距离选 ρ_i|N_i| 个最近邻
      - 阶段2：在本地随机 batch 上，选 loss ≤ 自身 的邻居；若空则选最优单邻居
      - 聚合：对选中的邻居参数做“算术平均”
      - 写回：x_new = α * x_self + (1-α) * average
    返回: global_loss, losses(list)
    """
    num_nodes = config.num_nodes
    device    = config.device

    # 超参数（可通过 config 覆盖）
    rho   = float(getattr(config, "ubar_rho", 0.4))   # 候选比例 ρ_i
    alpha = float(getattr(config, "ubar_alpha", 0.5)) # GUF 混合系数 α

    # -------- 本地训练（无 TV；penalty 只占位）--------
    snapshots = [[p.detach().cpu().clone() for p in m.parameters()] for m in models]
    zero_penalties = [[torch.zeros_like(p) for p in snapshots[i]] for i in range(num_nodes)]

    trained_params = [None] * num_nodes  # CPU param list
    losses         = [float('nan')] * num_nodes

    for i in range(num_nodes):
        n, l, tparams = _train_one_node(
            i, models[i], trainloaders[i], device,
            criterion, current_lr, config.local_epochs,
            penalty=[q.to(device) for q in zero_penalties[i]],
        )
        losses[n]         = l
        trained_params[n] = tparams

    # -------- 广播参数 & （可选）拜占庭攻击注入 --------
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
            # 广播用（CPU）
            params_to_send[byz] = [p.detach().cpu().clone() for p in fake]
            # 本地模型写入伪参数，影响下一轮快照
            with torch.no_grad():
                for pm, pf in zip(models[byz].parameters(), fake):
                    pm.data.copy_(pf)

    # -------- 小工具 --------
    def _l2(xa: List[torch.Tensor], xb: List[torch.Tensor]) -> float:
        s = 0.0
        for pa, pb in zip(xa, xb):
            d = (pa - pb).view(-1)
            s += float(torch.dot(d, d).item())
        return s ** 0.5

    def _average(params_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """对若干‘参数列表’做逐张量算术平均（均在 CPU）"""
        if len(params_list) == 1:
            return [p.clone() for p in params_list[0]]
        out = [torch.zeros_like(p) for p in params_list[0]]
        with torch.no_grad():
            for plist in params_list:
                for k, pk in enumerate(plist):
                    out[k].add_(pk)
            for k in range(len(out)):
                out[k].div_(len(params_list))
        return out

    def _set_model_params_(model: nn.Module, plist: List[torch.Tensor], dev: torch.device):
        """把 CPU 参数列表灌到模型（in-place, no grad）"""
        with torch.no_grad():
            for pm, psrc in zip(model.parameters(), plist):
                pm.data.copy_(psrc.to(dev))

    # -------- UBAR 两阶段聚合 + α 混合 --------
    new_params_cpu: List[List[torch.Tensor]] = [None] * num_nodes
    byz_set = set(byzantine_indices)

    for i in range(num_nodes):
        # 邻域 N_i（不含 self）
        neigh = [j for j, c in enumerate(adj_matrix[i]) if c]
        if len(neigh) == 0:
            # 没有邻居：仅保留自身
            new_params_cpu[i] = [p.clone() for p in trained_params[i]]
            continue

        # ---------- 阶段1：按距离筛 ρ|N_i| ----------
        k_cand = max(1, int(np.ceil(rho * len(neigh))))
        dists = []
        for j in neigh:
            dist_ij = _l2(trained_params[i], params_to_send[j])
            dists.append((j, dist_ij))
        dists.sort(key=lambda t: t[1])  # 从近到远
        Ns = [j for j, _ in dists[:k_cand]]  # 候选集合

        # ---------- 阶段2：性能筛选（用本地一个随机 batch） ----------
        # 取一个 batch
        try:
            it = iter(trainloaders[i])
            batch = next(it)
        except StopIteration:
            it = iter(trainloaders[i])
            batch = next(it)
        data, target = batch
        data, target = data.to(device), target.to(device)

        # 复用同一个模型结构测 loss（eval 模式、no_grad）
        probe_model = copy.deepcopy(models[i]).to(device)
        probe_model.eval()

        # 自身 loss
        _set_model_params_(probe_model, trained_params[i], device)
        with torch.no_grad():
            out_i = probe_model(data)
            loss_i = float(criterion(out_i, target).item())

        # 候选邻居的 loss
        cand_losses = []
        with torch.no_grad():
            for j in Ns:
                _set_model_params_(probe_model, params_to_send[j], device)
                out_j = probe_model(data)
                loss_j = float(criterion(out_j, target).item())
                cand_losses.append((j, loss_j))

        # 形成 Nr：loss_j <= loss_i 的邻居；若空则取最优单邻居
        Nr = [j for j, lj in cand_losses if lj <= loss_i]
        if len(Nr) == 0:
            j_star = min(cand_losses, key=lambda t: t[1])[0]
            Nr = [j_star]

        # ---------- 聚合：算术平均 ----------
        picked_params = [params_to_send[j] for j in Nr]
        R_i = _average(picked_params)  # CPU 列表

        # ---------- α 混合：x_new = α x_i + (1-α) R_i ----------
        x_i = trained_params[i]
        mixed = [alpha * xi + (1.0 - alpha) * ri for xi, ri in zip(x_i, R_i)]
        new_params_cpu[i] = mixed

    # -------- 回写到设备 --------
    for i in range(num_nodes):
        with torch.no_grad():
            for pm, pnew in zip(models[i].parameters(), new_params_cpu[i]):
                pm.data.copy_(pnew.to(device))

    global_loss = float(np.nanmean(losses))
    return global_loss, losses
