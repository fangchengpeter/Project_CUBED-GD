# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
import os
import time

config = Config()

class SimpleCNN(nn.Module):  # Actually a simple linear classifier for MNIST
    def __init__(self, seed=42):
        super(SimpleCNN, self).__init__()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(28 * 28, 10)  # Linear classifier

        if config.init_method == 'random':
            torch.manual_seed(seed)
            self._initialize_weights()
        elif config.init_method == 'remote':
            remote_state = config.remote_state_path
            if os.path.exists(remote_state):
                self._init_remote(remote_state)
            else:
                raise FileNotFoundError(f"Remote state file {remote_state} does not exist.")

    def _initialize_weights(self):
        # 1. Save the current torch RNG state.
        rng_state = torch.get_rng_state()
        # 2. Perturb the seed with real entropy.
        true_seed = int.from_bytes(os.urandom(8), 'little') ^ int(time.time() * 1e6)
        torch.manual_seed(true_seed)

        # 3. Initialize parameters.
        nn.init.normal_(self.fc.weight, mean=config.mean, std=config.std)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

        # 4. Restore the previous global RNG state.
        torch.set_rng_state(rng_state)

    def _init_remote(self, remote_state):
        state_to_load = torch.load("new_init_weight.pt")  # Directly a tensor.
        assert state_to_load.shape == self.fc.weight.shape, \
            f"Loaded weight shape {state_to_load.shape} does not match {self.fc.weight.shape}"
        self.fc.weight.data.copy_(state_to_load)   

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x



class CIFAR10CNN(nn.Module): 
    def __init__(self, seed=42):
        super(CIFAR10CNN, self).__init__()
        torch.manual_seed(seed)
        self.droprate = 0.25

        def conv_block(in_channels, out_channels, groups=8):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=groups, num_channels=out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=groups, num_channels=out_channels),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(self.droprate)
            )

        self.block1 = conv_block(3, 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.dropout = nn.Dropout(self.droprate)
        self.fc2 = nn.Linear(128, 10)

        self._initialize_weights()

    def _initialize_weights(self):
        # 1. Save the current torch RNG state.
        rng_state = torch.get_rng_state()
        # 2. Perturb the seed with real entropy.
        true_seed = int.from_bytes(os.urandom(8), 'little') ^ int(time.time() * 1e6)
        torch.manual_seed(true_seed)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=config.std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 4. Restore the previous global RNG state.
        torch.set_rng_state(rng_state)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MiniViT(nn.Module):
    def __init__(
        self,
        seed=42,
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        torch.manual_seed(seed)

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        rng_state = torch.get_rng_state()
        true_seed = int.from_bytes(os.urandom(8), "little") ^ int(time.time() * 1e6)
        torch.manual_seed(true_seed)

        nn.init.normal_(self.patch_embed.weight, mean=0.0, std=config.std)
        if self.patch_embed.bias is not None:
            nn.init.constant_(self.patch_embed.bias, 0)

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=config.std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        torch.set_rng_state(rng_state)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.encoder(x)
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x



def node_seed(base_seed: int, node_id: int) -> int:
    """get a seed from base_seed and node id."""
    return base_seed + node_id * 1000 + 12345

class LogisticRegression(nn.Module):
    def __init__(
        self,
        seed: int = 42,                # base_seed
        node_id: int = 0,              # node (0..num_nodes-1)
        input_dim: int = 50,
        wstar_path: str = "wstar.pt",
        perturb_mean: float = config.mean,
        perturb_std:  float = config.std,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim, 1)
        self.to(device)

        self._initialize_weights(
            base_seed=seed,
            node_id=node_id,
            wstar_path=wstar_path,
            perturb_mean=perturb_mean,
            perturb_std=perturb_std,
            device=device,
        )

    @torch.no_grad()
    def _initialize_weights(
        self,
        base_seed: int,
        node_id: int,
        wstar_path: str,
        perturb_mean: float,
        perturb_std: float,
        device: str | torch.device,
    ):
        # 1) load w*, b*
        ckpt = torch.load(wstar_path, map_location="cpu")
        self.fc.weight.data.copy_(ckpt["w_star"].to(device))
        self.fc.bias.data.copy_(ckpt["b_star"].to(device))

        # 2) generate a node-specific seed for perturbation 
        # each reproduce will be the same, but different nodes get different inits
        seed_value = node_seed(base_seed, node_id)
        gen = torch.Generator(device="cpu").manual_seed(seed_value)

        # 3) sample reproducible node-level perturbations and add them to the parameters
        if perturb_std > 0:
            w_noise = torch.normal(
                mean=perturb_mean,
                std=perturb_std,
                size=self.fc.weight.shape,
                generator=gen,
            ).to(device)
            b_noise = torch.normal(
                mean=perturb_mean,
                std=perturb_std,
                size=self.fc.bias.shape,
                generator=gen,
            ).to(device)
            self.fc.weight.add_(w_noise)
            self.fc.bias.add_(b_noise)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x.squeeze(-1)
