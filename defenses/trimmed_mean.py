# defenses/trimmed_mean.py
import torch
from typing import List
from .base import BaseDefense

class TrimmedMean(BaseDefense):
    def __call__(self, params_list: List[List[torch.Tensor]],
                   num_byzantine_expected: int, # This is 'b' or 'f'
                   device: torch.device,
                   **kwargs) -> List[torch.Tensor]:
        """
        Coordinate-wise trimmed mean. num_byzantine_expected is used as the trim_param.
        """
        # trim_param typically config.max_byzantine_nodes, passed as num_byzantine_expected
        # Or, if defense needs its own specific trim param different from global 'f':
        # trim_param = self.config.trim_parameter if self.config else num_byzantine_expected
        trim_param = num_byzantine_expected # Assuming it's the 'f' to trim

        # It's common to also include the node's own parameters in the list for trimmed mean
        current_params = kwargs.get('current_params', None)
        all_params_for_trim = []
        if current_params:
            all_params_for_trim.append(current_params)
        for p_set in params_list:
            all_params_for_trim.append(p_set)

        if not all_params_for_trim:
             raise ValueError("No parameters to aggregate in TrimmedMean.")

        if len(all_params_for_trim) <= 2 * trim_param:
            # Fallback or error: Not enough parameters to trim 'trim_param' from both ends.
            # Option 1: Raise error
            raise ValueError(
                f"Insufficient parameters for trimmed mean. "
                f"Need more than {2 * trim_param} parameter sets, but only have {len(all_params_for_trim)}."
            )
            # Option 2: Fallback to simple mean (less robust)
            # print(f"Warning: Not enough params for trimmed mean, falling back to simple average.")
            # return NoDefense(self.config)(params_list, num_byzantine_expected, device, **kwargs)


        num_layers = len(all_params_for_trim[0])
        aggregated_params = []

        for layer_idx in range(num_layers):
            original_shape = all_params_for_trim[0][layer_idx].shape
            # Ensure all params for this layer are on the correct device
            layer_tensors = [p_set[layer_idx].to(device).reshape(-1) for p_set in all_params_for_trim]
            param_values_flat = torch.stack(layer_tensors, dim=0) # Shape: (num_models, flattened_layer_size)

            sorted_values, _ = torch.sort(param_values_flat, dim=0)

            if trim_param > 0:
                trimmed_values = sorted_values[trim_param:-trim_param]
            else: # No trimming
                trimmed_values = sorted_values

            aggregated_param_flat = torch.mean(trimmed_values, dim=0, dtype=torch.float)
            aggregated_params.append(aggregated_param_flat.reshape(original_shape))

        return aggregated_params