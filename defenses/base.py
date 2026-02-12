# defenses/base.py
from abc import ABC, abstractmethod
from typing import List
import torch

class BaseDefense(ABC):
    """
    Abstract base class for all defense/aggregation algorithms.
    """
    def __init__(self, config=None):
        """
        Initialize the defense mechanism.
        Args:
            config (Config, optional): Configuration object if needed for defense parameters.
        """
        self.config = config

    @abstractmethod # __call__ is THE abstract method to implement
    def __call__(self, params_list: List[List[torch.Tensor]],
                   num_byzantine_expected: int,
                   device: torch.device,
                   **kwargs) -> List[torch.Tensor]:
        """
        Execute the defense/aggregation mechanism.

        Args:
            params_list (List[List[torch.Tensor]]):
                A list of parameter sets. Each parameter set is a list of tensors
                (e.g., parameters received from neighboring nodes).
            num_byzantine_expected (int):
                The number of Byzantine nodes the defense should assume/tolerate.
            device (torch.device):
                The device for computations (e.g., 'cuda' or 'cpu').
            **kwargs:
                Additional keyword arguments that specific defenses might require.
                Example: current_params=node_own_locally_trained_params (List[torch.Tensor])

        Returns:
            List[torch.Tensor]: The aggregated/selected model parameters (as a list of tensors).
        """
        pass