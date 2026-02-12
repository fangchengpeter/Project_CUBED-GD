import time
import copy
from typing import List, Optional
import numpy as np
import os
import torch
import torch.nn as nn

from config import Config

try:
    from defenses import BaseDefense
except ImportError:
    BaseDefense = object

from byzantine import get_byzantine_params
import concurrent.futures
import os

def _train_one_node(node_idx, model, trainloader, device, criterion, current_lr, local_epochs):
    model_copy = copy.deepcopy(model).to(device)
    model_copy.train()
    lamb = 0.001

    last_epoch_losses = []

    for epoch_idx in range(local_epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            # print(f"Node {node_idx} saw {data.shape[0]} samples")
            
            output = model_copy(data)
            # print("Output NaN?", torch.isnan(output).any().item())
            l2_reg = sum((p ** 2).sum() for p in model_copy.parameters())
            loss = criterion(output, target) + lamb * l2_reg
            model_copy.zero_grad()
            loss.backward()

            # if node_idx == 0:
            #     total_weight_norm = 0.0
            #     total_grad_norm = 0.0
            #     for name, param in model_copy.named_parameters():
            #         with torch.no_grad():
            #             total_weight_norm += param.norm().item() ** 2
            #             if param.grad is not None:
            #                 total_grad_norm += param.grad.norm().item() ** 2

            #     total_weight_norm = total_weight_norm ** 0.5
            #     total_grad_norm = total_grad_norm ** 0.5
            #     print(f"[Node {node_idx}] Total weight L2 norm: {total_weight_norm:.6f}, total grad L2 norm: {total_grad_norm:.6f}")

                    

            with torch.no_grad():
                for param in model_copy.parameters():
                    if param.grad is not None:
                        param.data -= current_lr * param.grad

            if epoch_idx == local_epochs - 1:
                last_epoch_losses.append(loss.item())

    epoch_loss = float(np.mean(last_epoch_losses)) if last_epoch_losses else float('nan')
    trained_params = [p.data.clone() for p in model_copy.parameters()]
    return node_idx, epoch_loss, trained_params


def train_epoch(models: List[nn.Module],
                trainloaders: List[torch.utils.data.DataLoader],
                adj_matrix: np.ndarray,
                byzantine_indices: List[int],
                criterion: nn.Module,
                current_lr: float,
                defense_obj: Optional[BaseDefense],
                attack_type: str,
                config,
                epoch: int) -> (float, List[float]):

    num_nodes = config.num_nodes
    device = config.device

    locally_trained_model_params_list = [None] * num_nodes
    epoch_losses = [float('nan')] * num_nodes

    for i in range(num_nodes):
        node_idx, avg_loss, trained_params = _train_one_node(
            i, models[i], trainloaders[i], device, criterion, current_lr, config.local_epochs
        )
        epoch_losses[node_idx] = avg_loss
        locally_trained_model_params_list[node_idx] = trained_params



    # 注入 Byzantine 攻击
    params_to_be_sent = locally_trained_model_params_list.copy()
    if attack_type and attack_type.lower() != "none":
        honest_indices = [i for i in range(num_nodes) if i not in byzantine_indices]
        honest_params = [locally_trained_model_params_list[i]
                         for i in honest_indices
                         if locally_trained_model_params_list[i] is not None]
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

    # >>> modified for multiple consensus <<<
    for _ in range(config.local_consensus):
        new_model_params = []

        for node_idx in range(num_nodes):
            neighbors = [j for j, conn in enumerate(adj_matrix[node_idx]) if conn]
            received = [params_to_be_sent[j]
                        for j in neighbors
                        if 0 <= j < num_nodes and params_to_be_sent[j] is not None]
            current = params_to_be_sent[node_idx]

            if defense_obj:
                try:
                    agg = defense_obj(
                        params_list=received,
                        num_byzantine_expected=config.expected_byzantine_nodes,
                        device=device,
                        current_params=current
                    )
                except Exception as e:
                    print(f"Defense aggregation failed for node {node_idx}: {e}")
                    agg = current
            else:
                to_avg = [current] + received
                if to_avg:
                    layers = len(to_avg[0])
                    agg = []
                    for i in range(layers):
                        stacked = torch.stack([p[i] for p in to_avg], dim=0)
                        agg.append(torch.mean(stacked, dim=0))
                else:
                    agg = current

            new_model_params.append(agg)

        params_to_be_sent = new_model_params  # 更新用于下一轮聚合的模型参数

    # >>> apply updated model to real model parameters <<<
    for node_idx in range(num_nodes):
        for param, new_val in zip(models[node_idx].parameters(), params_to_be_sent[node_idx]):
            param.data = new_val.clone()

    # 计算平均 loss
    honest_losses = [l for i, l in enumerate(epoch_losses)
                     if i not in byzantine_indices and not np.isnan(l)]
    mean_loss = np.mean(honest_losses) if honest_losses else float('inf')
    return mean_loss, epoch_losses

def _evaluate_one_node(node_idx, model, testloader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()

    acc = 100 * correct / total if total > 0 else 0.0
    return node_idx, acc


def evaluate_models(models: List[nn.Module],
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
