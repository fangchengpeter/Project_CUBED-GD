# config.py
import torch
import os
import random
import numpy as np
import math
from datetime import datetime


class Config:
    def __init__(self):
        # Hyperparameters
        self.dataset = "cifar10"  # Options: "mnist", "cifar10", "synthetic"
        self.model = "vit"  # Options for vision datasets: "cnn", "vit"
        self.num_nodes = 10
        self.max_byzantine_nodes = 2 #f
        self.expected_byzantine_nodes = 2 #b
        self.learning_rate = 0.06

        # Adjust batch_size calculation based on dataset
        if self.dataset == "mnist":
            total_train_samples = 60000
        elif self.dataset == "cifar10":
            total_train_samples = 50000
        elif self.dataset == "synthetic":
            total_train_samples = 60000
        else:
            raise ValueError(f"Unsupported dataset in Config: {self.dataset}")
        self.loader_batch_size = int(total_train_samples // self.num_nodes)  # Per-node batch size for one pass
        if self.model == "vit":
            self.loader_batch_size = 64
        self.local_epochs = 1
        self.local_consensus = 1

        self.num_epochs = 1500
        self.plot_interval = 20
        self.connectivity = 0.5
        self.seed = 42
        self.variant = "trimmed_mean"  
        self.std = 0
        self.mean = 0
        self.init_method = 'random'  # Options: 'random', 'remote'
        self.remote_state_path = "new_init_weight.pt"  # Path to the remote state file

        # Attack parameters
        self.attack_type = "random"  # Options: "none", "random", "sign_flipping", "scaled", "label_flipping", "
        self.random_attack_var = 10 # Reduced from 10, which might be too large
        self.scaled_attack_scale = 10.0  # For scaled attack
        self.label_flipping_source_label = 0  # For label_flipping attack
        self.label_flipping_target_label = 1  # For label_flipping attack
        self.constant_attack_value = 100.0  # For constant attack

        # Paths
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.data_dir = './data'  # Centralized data directory
        os.makedirs(self.data_dir, exist_ok=True)

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Keep as CPU for now given previous issues
        print(f"Using device: {self.device}")

        # Set random seeds for reproducibility
        self.set_seeds()

    def set_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():  # This check is fine even if device is CPU
            torch.cuda.manual_seed_all(self.seed)  # Use manual_seed_all for multi-GPU consistency

    def lr_schedule(self, epoch):
        return self.learning_rate / (1 + 0.01 * epoch)


    def save_config(self):
        """Save configuration to a file"""
        # self.result_dir needs to be set by main.py before calling this
        if not hasattr(self, 'result_dir') or not self.result_dir:
            print("Warning: config.result_dir not set. Cannot save config.")
            return

        config_path = os.path.join(self.result_dir, "config.txt")
        try:
            with open(config_path, 'w') as f:
                sorted_config = dict(sorted(self.__dict__.items()))
                for key, value in sorted_config.items():
                    # Avoid trying to write complex objects like torch.device directly
                    if isinstance(value, torch.device):
                        f.write(f"{key}: {str(value)}\n")
                    elif key != 'result_dir':  # Avoid self-reference if result_dir is part of config by now
                        f.write(f"{key}: {value}\n")
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving config: {e}")
