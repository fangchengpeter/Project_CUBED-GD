import torch
from typing import List
from .base import BaseDefense

class GeoMedian(BaseDefense):
    """
    GeoMedian defense using full-model flattening.
    All layers' parameters are concatenated and aggregated together.
    """

    def __call__(self, params_list: List[List[torch.Tensor]],
                 num_byzantine_expected: int,
                 device: torch.device,
                 **kwargs) -> List[torch.Tensor]:
        
        trim_param = num_byzantine_expected  # Currently unused
        current_params = kwargs.get('current_params', None)

        # DO NOT CHANGE: keep original param preparation
        all_params_for_median = []
        if current_params:
            all_params_for_median.append(current_params)
        for p_set in params_list:
            all_params_for_median.append(p_set)

        if not all_params_for_median:
            raise ValueError("No parameters to aggregate in Median.")

        # Step 1: flatten all parameters for each client
        flat_params_list = []
        for param_set in all_params_for_median:
            flat_tensor = torch.cat([p.to(device).reshape(-1) for p in param_set], dim=0)
            flat_params_list.append(flat_tensor)
        
        stacked = torch.stack(flat_params_list, dim=0)  # shape: (num_clients, total_params)
        median_flat = self._geometric_median(stacked)

        # Step 2: reshape back to per-layer tensors
        shapes = [p.shape for p in all_params_for_median[0]]
        numels = [torch.tensor(s).prod().item() for s in shapes]

        split_params = torch.split(median_flat, numels)
        aggregated_params = [param.reshape(shape) for param, shape in zip(split_params, shapes)]

        return aggregated_params

    def _geometric_median(self, X: torch.Tensor, eps: float = 1e-5, max_iter: int = 100) -> torch.Tensor:
        """
        Compute geometric median using Weiszfeld’s algorithm.
        X: Tensor of shape (n_points, dim)
        Returns: Tensor of shape (dim,)
        """
        guess = torch.mean(X, dim=0)
        for _ in range(max_iter):
            diff = X - guess
            dist = torch.norm(diff, dim=1).clamp(min=1e-8)
            weights = 1.0 / dist
            new_guess = torch.sum(weights[:, None] * X, dim=0) / weights.sum()
            if torch.norm(new_guess - guess) < eps:
                break
            guess = new_guess
        return guess
