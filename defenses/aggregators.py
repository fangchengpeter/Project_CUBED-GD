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
            config (Config, optional): Configuration object from main.py,
                                       if the defense needs specific parameters from it.
        """
        self.config = config # Allows defenses to access config.trim_parameter etc.

    @abstractmethod
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
                The number of Byzantine nodes the defense should assume/tolerate (e.g., 'f' in Krum).
                This usually comes from `config.max_byzantine_nodes`.
            device (torch.device):
                The device for computations (e.g., 'cuda' or 'cpu').
            **kwargs:
                Additional keyword arguments that specific defenses might require.
                For example, a defense might need the current node's own parameters
                if it's doing a weighted average or comparison: `current_params=...`.

        Returns:
            List[torch.Tensor]: The aggregated/selected model parameters (as a list of tensors).
        """
        pass