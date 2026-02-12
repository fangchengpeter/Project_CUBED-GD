# defenses/krum.py
import torch
from typing import List
from .base import BaseDefense

class Krum(BaseDefense):
    def __call__(self, params_list: List[List[torch.Tensor]],
                   num_byzantine_expected: int, # This is 'f'
                   device: torch.device,
                   **kwargs) -> List[torch.Tensor]:
        """
        Krum screening. Selects one model's parameters.
        params_list should contain all candidates (e.g., self + neighbors).
        """
        current_params = kwargs.get('current_params', None)
        all_params_for_krum = []
        if current_params:
            all_params_for_krum.append(current_params)
        for p_set in params_list: # params_list are from neighbors
            all_params_for_krum.append(p_set)

        if not all_params_for_krum:
             raise ValueError("No parameters to aggregate in Krum.")

        num_candidates = len(all_params_for_krum)
        # Number of "closest" models to sum distances for, excluding self
        # In Krum, we need at least f+1 honest nodes, so n >= 2f+3 or n > 2f+2 to select one.
        # The number of neighbors to select is n - f - 2 (excluding self, so n-1 other models)
        # If we sum distances to k closest, k = n_candidates - num_byzantine_expected - 2
        num_to_select_closest = num_candidates - num_byzantine_expected - 2

        if num_to_select_closest < 0: # Standard Krum requires at least 1 (k=0 means sum 0 dist)
             raise ValueError(
                f"Insufficient candidates for Krum. Need at least {num_byzantine_expected + 3} "
                f"candidates (self + neighbors), but only have {num_candidates}."
            )
        # If num_to_select_closest is 0, it means we consider distance to 0 other nodes (score is 0).
        # This happens if num_candidates = num_byzantine_expected + 2.
        # Usually Krum needs n > 2f + 2 => n >= 2f + 3. So num_candidates >= 2f+3.
        # num_to_select_closest = (num_candidates -1) - num_byzantine_expected -1  (as per original Krum with n-f-1 others)

        # Flatten all parameter sets for distance calculation
        flat_param_sets = []
        for p_set in all_params_for_krum:
            # Ensure all tensors in p_set are on the correct device
            flat_layers = [p_layer.to(device).reshape(-1) for p_layer in p_set]
            flat_param_sets.append(torch.cat(flat_layers))

        # Calculate pairwise squared Euclidean distances
        distances_sq = torch.zeros((num_candidates, num_candidates), device=device)
        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                dist_sq = torch.sum((flat_param_sets[i] - flat_param_sets[j]) ** 2)
                distances_sq[i, j] = distances_sq[j, i] = dist_sq

        scores = torch.zeros(num_candidates, device=device)
        for i in range(num_candidates):
            # Sort distances to other models
            sorted_dists_sq_for_i, _ = torch.sort(distances_sq[i])
            # Sum distances to the (num_candidates - num_byzantine_expected - 2) closest models
            # The first element is dist to self (0), so we take from 1 up to num_to_select_closest + 1
            if num_to_select_closest > 0 :
                scores[i] = torch.sum(sorted_dists_sq_for_i[1 : num_to_select_closest + 1])
            else: # if num_to_select_closest is 0 or less (e.g. n = f+2), score is 0 for all. Pick first.
                scores[i] = 0


        selected_index = torch.argmin(scores).item()
        return all_params_for_krum[selected_index] # Return the parameters of the selected model