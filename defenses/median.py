# defenses/median.py
import torch
from typing import List
from .base import BaseDefense

class Median(BaseDefense):
    def __call__(self, params_list: List[List[torch.Tensor]],
                   num_byzantine_expected: int,
                   device: torch.device,
                   **kwargs) -> List[torch.Tensor]:
        """
        Coordinate-wise median.
        """
        trim_param = num_byzantine_expected  # Assuming it's the 'f' to trim
        current_params = kwargs.get('current_params', None)
        all_params_for_median = []
        if current_params:
            all_params_for_median.append(current_params)
        for p_set in params_list:
            all_params_for_median.append(p_set)

        if not all_params_for_median:
             raise ValueError("No parameters to aggregate in Median.")

        if len(all_params_for_median) <= 2 * trim_param:
            # Fallback or error: Not enough parameters to trim 'trim_param' from both ends.
            # Option 1: Raise error
            raise ValueError(
                f"Insufficient parameters for trimmed mean. "
                f"Need more than {2 * trim_param} parameter sets, but only have {len(all_params_for_trim)}."
            )
        num_layers = len(all_params_for_median[0])
        aggregated_params = []

        for layer_idx in range(num_layers):
            original_shape = all_params_for_median[0][layer_idx].shape
            layer_tensors = [p_set[layer_idx].to(device).reshape(-1) for p_set in all_params_for_median]
            param_values_flat = torch.stack(layer_tensors, dim=0)
            sorted_values, _ = torch.sort(param_values_flat, dim=0)
            if trim_param > 0:
                trimmed_values = sorted_values[trim_param:-trim_param]
            else:
                trimmed_values = sorted_values
            median_values_flat, _ = torch.median(trimmed_values, dim=0)
            aggregated_params.append(median_values_flat.reshape(original_shape))

        return aggregated_params