# defenses/bridge_b.py
import torch
from typing import List
from .base import BaseDefense
from .krum import Krum # Assuming Krum class is defined
from .trimmed_mean import TrimmedMean # Assuming TrimmedMean class is defined
from .median import Median # Fallback

class BridgeB(BaseDefense):
    def __call__(self, params_list: List[List[torch.Tensor]],
                   num_byzantine_expected: int, # This is 'f' or 'b'
                   device: torch.device,
                   **kwargs) -> List[torch.Tensor]:
        """
        BRIDGE-B: Krum followed by Trimmed Mean.
        params_list should contain all candidates (e.g., self + neighbors).
        """
        current_params = kwargs.get('current_params', None)
        all_params_for_bridgeb = []
        if current_params:
            all_params_for_bridgeb.append(current_params)
        for p_set in params_list:
            all_params_for_bridgeb.append(p_set)

        if not all_params_for_bridgeb:
             raise ValueError("No parameters to aggregate in BridgeB.")

        num_candidates = len(all_params_for_bridgeb)
        f = num_byzantine_expected

        # Condition from original paper for Multi-Krum might be n >= 2f+1 to select n-2f models
        # Here, we select (num_candidates - 2*f) models using recursive Krum
        num_to_select_by_krum = num_candidates - 2 * f

        # Check if we have enough neighbors for the algorithm (rough condition)
        # Krum itself needs num_remaining >= f + 3 (approx) at each step.
        # Trimmed mean needs len(selected_krum_params) > 2 * f (if f is also trim param for inner TM)
        if num_candidates < 2 * f + 3: # Need at least one model after Krum selection for TM
             raise ValueError(
                f"Insufficient candidates for BridgeB Krum stage. "
                f"Need at least {2 * f + 3} candidates, got {num_candidates}."
            )
        if num_to_select_by_krum <= 0:
            raise ValueError(
                f"BridgeB: num_to_select_by_krum is {num_to_select_by_krum}, must be > 0."
            )


        # Instantiate Krum for internal use (it needs a config if Krum itself is configurable)
        # For simplicity, assume Krum() can be called without a config or uses a default one
        internal_krum = Krum(config=self.config) # Pass config if Krum needs it

        selected_by_krum_params = []
        remaining_params_for_krum = list(all_params_for_bridgeb) # Make a mutable copy

        for _ in range(num_to_select_by_krum):
            if not remaining_params_for_krum: break
            try:
                # Krum selects ONE best model from the current `remaining_params_for_krum`
                # The `num_byzantine_expected` for this internal Krum call is still `f`,
                # as we assume `f` attackers are present in any subset of sufficient size.
                krum_selected_model_params = internal_krum(
                    remaining_params_for_krum,
                    num_byzantine_expected=f,
                    device=device
                    # No current_params for internal Krum as it operates on the list given
                )
                selected_by_krum_params.append(krum_selected_model_params)

                # Remove the selected one from remaining_params_for_krum
                # This removal needs to be careful due to comparing lists of tensors.
                # A more robust way is to track indices or use a unique ID if available.
                # For now, find by object identity or deep equality if params are unique.
                # This is a bit inefficient.
                for idx, p_set in enumerate(remaining_params_for_krum):
                    is_match = True
                    if len(p_set) != len(krum_selected_model_params):
                        is_match = False
                    else:
                        for l_idx in range(len(p_set)):
                            if not torch.equal(p_set[l_idx], krum_selected_model_params[l_idx]):
                                is_match = False
                                break
                    if is_match:
                        remaining_params_for_krum.pop(idx)
                        break
            except ValueError as e: # Krum might raise ValueError if not enough candidates
                print(f"BridgeB: Krum selection failed internally, not enough candidates left. {e}")
                break # Stop Krum selection phase

        if not selected_by_krum_params:
            print("BridgeB: Krum selection yielded no parameters. Falling back to Median.")
            return Median(self.config)(all_params_for_bridgeb, f, device, **kwargs)

        # Apply Trimmed Mean on the Krum-selected parameters
        # The trim parameter for this stage in Bridge-B is also 'f'
        internal_trimmed_mean = TrimmedMean(config=self.config)
        try:
            final_aggregated_params = internal_trimmed_mean(
                selected_by_krum_params, # List of params selected by Krum
                num_byzantine_expected=f, # Trim 'f' from each end
                device=device
                # No current_params for internal TM, it operates on the list given
            )
        except ValueError as e:
            print(f"BridgeB: Trimmed Mean stage failed: {e}. Falling back to Median on Krum selected params.")
            return Median(self.config)(selected_by_krum_params, f, device, **kwargs)

        return final_aggregated_params