# train.py
import copy
from typing import List, Optional, Tuple
import numpy as np
import os
import torch
import torch.nn as nn

try:
    from defenses import BaseDefense  # 需要你工程中已有
except ImportError:
    BaseDefense = object

from byzantine import get_byzantine_params  # 需要你工程中已有





def _train_one_node(
    node_idx: int,
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    device: torch.device | str,
    criterion: nn.Module,
    current_lr: float,
    local_epochs: int,
    l2_lambda: float = 0.01,
) -> Tuple[int, float, List[torch.Tensor]]:
    """
    单节点本地训练（二分类 sigmoid）：
      - 模型输出单 logit: output shape [B] 或 [B,1]
      - 损失：nn.BCEWithLogitsLoss()
      - 更新：手写 SGD
    返回：节点索引、本地最后一轮的平均 BCE loss、训练后的参数列表
    """
    model_copy = copy.deepcopy(model).to(device)
    model_copy.train()

    criterion = nn.BCEWithLogitsLoss()

    last_epoch_losses: List[float] = []

    for epoch_idx in range(local_epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)

            logits = model_copy(data)
            
            # print(target_f.shape, logits.shape, target.shape)

            loss_main = criterion(logits, target.float())
            l2_reg = 0.0
            for param in model_copy.parameters():
                l2_reg += torch.norm(param, p=2) ** 2
            loss = loss_main + l2_lambda * l2_reg


            # 反传
            for p in model_copy.parameters():
                p.grad = None
            loss.backward()

            #更新GD
            with torch.no_grad():
                for p in model_copy.parameters():
                    p -= current_lr * p.grad
            # 只记录纯 BCE（便于观察分类损失）
            if epoch_idx == local_epochs - 1:
                last_epoch_losses.append(loss_main.item())

    epoch_loss = float(np.mean(last_epoch_losses)) if last_epoch_losses else float('nan')
    trained_params = [p.data.clone() for p in model_copy.parameters()]
    return node_idx, epoch_loss, trained_params


def synthetic_train_epoch(
    models: List[nn.Module],
    trainloaders: List[torch.utils.data.DataLoader],
    adj_matrix: np.ndarray,
    byzantine_indices: List[int],
    criterion: nn.Module,
    current_lr: float,
    defense_obj: Optional[BaseDefense],
    attack_type: str,
    config,
    epoch: int
) -> Tuple[float, List[float]]:
    """
    单轮联邦训练（Sigmoid 二分类）：
      1) 各节点本地训练（手写 SGD）
      2) 可选 Byzantine 攻击注入
      3) 多次局部共识聚合（config.local_consensus）
      4) 回写聚合后的参数到各节点模型
    返回：诚实节点 loss 的均值（最后一轮本地 BCE）、每节点 loss 列表
    """
    num_nodes = config.num_nodes
    device = config.device

    # L2 系数优先从 config 读取（若无则用默认 0.01）
    l2_lambda = getattr(config, "l2_lambda", 0.01)

    locally_trained_model_params_list: List[Optional[List[torch.Tensor]]] = [None] * num_nodes
    epoch_losses: List[float] = [float('nan')] * num_nodes

    # 1) 各节点本地训练
    for i in range(num_nodes):
        node_idx, avg_loss, trained_params = _train_one_node(
            i, models[i], trainloaders[i], device, criterion, current_lr, config.local_epochs, l2_lambda
        )
        epoch_losses[node_idx] = avg_loss
        locally_trained_model_params_list[node_idx] = trained_params

    # 2) 攻击注入（可选）
    params_to_be_sent = locally_trained_model_params_list.copy()
    if attack_type and attack_type.lower() != "none":
        honest_indices = [i for i in range(num_nodes) if i not in byzantine_indices]
        honest_params = [
            locally_trained_model_params_list[i]
            for i in honest_indices
            if locally_trained_model_params_list[i] is not None
        ]
        for byz in byzantine_indices:
            if 0 <= byz < num_nodes and params_to_be_sent[byz] is not None:
                try:
                    params_to_be_sent[byz] = get_byzantine_params(
                        params_to_be_sent[byz],
                        attack_type,
                        honest_params=honest_params,
                        config=config,
                        device=device,
                    )
                except Exception as e:
                    print(f"Error applying attack for Byzantine node {byz} (epoch {epoch + 1}): {e}")

    # 3) 多次局部共识
    # params_to_be_sent: List[List[Tensor]]
    for _ in range(getattr(config, "local_consensus", 1)):
        new_model_params: List[List[torch.Tensor]] = []

        for node_idx in range(num_nodes):
            neighbors = [j for j, conn in enumerate(adj_matrix[node_idx]) if conn]
            received = [
                params_to_be_sent[j]
                for j in neighbors
                if 0 <= j < num_nodes and params_to_be_sent[j] is not None
            ]
            current = params_to_be_sent[node_idx]

            if defense_obj:
                try:
                    agg = defense_obj(
                        params_list=received,
                        num_byzantine_expected=getattr(config, "expected_byzantine_nodes", 0),
                        device=device,
                        current_params=current
                    )
                except Exception as e:
                    print(f"Defense aggregation failed for node {node_idx}: {e}")
                    agg = current
            else:
                to_avg = [current] + received if current is not None else received
                if to_avg:
                    layers = len(to_avg[0])
                    agg = []
                    for i_layer in range(layers):
                        stacked = torch.stack([p[i_layer] for p in to_avg], dim=0)
                        agg.append(torch.mean(stacked, dim=0))
                else:
                    agg = current

            new_model_params.append(agg)

        params_to_be_sent = new_model_params  # 下一轮共识的输入

    # 4) 回写聚合参数到真实模型
    for node_idx in range(num_nodes):
        if params_to_be_sent[node_idx] is None:
            continue
        for param, new_val in zip(models[node_idx].parameters(), params_to_be_sent[node_idx]):
            param.data = new_val.clone()

    # 计算诚实节点的平均 loss（最后一轮本地 BCE）
    honest_losses = [l for i, l in enumerate(epoch_losses) if i not in byzantine_indices and not np.isnan(l)]
    mean_loss = float(np.mean(honest_losses)) if honest_losses else float('inf')
    return mean_loss, epoch_losses

def _evaluate_one_node(node_idx, model, testloader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = (output >= 0).to(torch.long)  # Sigmoid 二分类阈值 0.0
            total += target.size(0)
            correct += (pred == target).sum().item()

    acc = 100 * correct / total if total > 0 else 0.0
    return node_idx, acc


def synthetic_evaluate_models(models: List[nn.Module],
                    testloader: torch.utils.data.DataLoader,
                    byzantine_indices: List[int],
                    config) -> (float, List[float]):

    device = config.device
    num_nodes = config.num_nodes
    node_accuracies = [float('nan')] * num_nodes
    honest_accs = []

    max_workers = min(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)), num_nodes)

    for idx in range(num_nodes):
        if idx not in byzantine_indices:
            node_idx, acc = _evaluate_one_node(idx, models[idx], testloader, device)
            node_accuracies[node_idx] = acc
            honest_accs.append(acc)

    mean_acc = float(np.nanmean(honest_accs)) if honest_accs else 0.0
    return mean_acc, node_accuracies
