# defenses/no_defense.py
import torch
from typing import List
from .base import BaseDefense

class NoDefense(BaseDefense):
    def __call__(self, params_list: List[List[torch.Tensor]],
                   num_byzantine_expected: int,
                   device: torch.device,
                   **kwargs) -> List[torch.Tensor]:
        """
        No screening, return the mean of all received parameters.
        If current_params are provided in kwargs, average them too (FedAvg style for a node).
        """
        current_params = kwargs.get('current_params', None)
        all_params_for_avg = []

        if current_params: # If node's own params are part of the averaging set
            all_params_for_avg.append(current_params)

        # Add parameters from neighbors
        for p_set in params_list:
            all_params_for_avg.append(p_set)

        if not all_params_for_avg:
            # This case should ideally be handled before calling a defense.
            # If a node has no neighbors and no current_params are passed, what to do?
            # For now, let's assume if current_params were needed, they were passed.
            # If params_list is empty and no current_params, and this is called, it's an issue.
            # Or, if only current_params are there, it means no neighbors, so it just uses its own.
            if current_params and not params_list:
                return current_params # No neighbors, use its own
            else: # Should not happen if called correctly by trainer
                raise ValueError("No parameters to aggregate in NoDefense.")


        num_layers = len(all_params_for_avg[0])
        aggregated_params = []

        for layer_idx in range(num_layers):
            stacked_layer = torch.stack([p_set[layer_idx].to(device) for p_set in all_params_for_avg], dim=0)
            avg_layer = torch.mean(stacked_layer, dim=0)
            aggregated_params.append(avg_layer)
        return aggregated_params