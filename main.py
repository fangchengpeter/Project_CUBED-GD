# main.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import time
import logging  # Use logging module
from typing import List, Optional  # For type hinting
from rtc import rtc_epoch
from secureDL import secure_epoch
from ubar import ubar_epoch
from braso import braso_epoch
from synthetic_train import synthetic_train_epoch, synthetic_evaluate_models
from config import Config
from data_loader import (
    load_data_without_validation,
    load_extreme_non_iid,
    load_moderate_non_iid,
    load_data_with_validation_split,
    load_moderate_non_iid_cifar10,
    load_extreme_non_iid_cifar10,
)
from models import SimpleCNN, CIFAR10CNN, LogisticRegression, MiniViT
from network import (
    create_adjacency_matrix,
    load_adjacency_matrix,
    select_byzantine_nodes,
    load_byzantine_nodes,
)
from train import train_epoch, evaluate_models
from analysis import run_analysis
from defenses import BaseDefense, NoDefense, TrimmedMean, Median, Krum, BridgeB, GeoMedian
from visualization import plot_adjacency_matrix

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFENSE_CLASS_MAP = {
    "none": NoDefense,
    "trimmed_mean": TrimmedMean,
    "median": Median,
    "krum": Krum,
    "bridge_b": BridgeB,
    "geo_median": GeoMedian,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Byzantine-Resilient Federated Learning (Project B Refactored)"
    )
    parser.add_argument(
        "--result_dir_suffix",
        type=str,
        default=None,
        help="Suffix for result directory name to distinguish runs",
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only analyze existing results from a specified result_dir",
    )
    parser.add_argument(
        "--result_dir_to_analyze",
        type=str,
        default=None,
        help="Full path to result_dir for analysis if --analyze_only is used",
    )
    parser.add_argument(
        "--defense_variant",
        type=str,
        default=None,
        help="Defense variant to use (overrides config.variant for defense selection)",
    )
    return parser.parse_args()

def save_average_distance(
    models: List[nn.Module],
    save_dir: str,
    summary_path: Optional[str] = None,
    w_star: Optional[object] = None,
) -> None:
    """
    Compute the mean L2 distance from each model to w* and save the average of the distance.
    Args:
        models: list of node models.
        save_dir: kept for backward compatibility; not used.
        summary_path: optional text file path to store the mean distance.
        w_star: optional module/state-dict/path representing the reference optimum.
    """

    if w_star is None:
        logger.warning("w_star not provided. Skipping average distance computation.")
        return

    if isinstance(w_star, str):
        if os.path.exists(w_star):
            w_star_state = torch.load(w_star, map_location="cpu")
            if isinstance(w_star_state, dict) and "model_state_dict" in w_star_state:
                w_star_params = w_star_state["model_state_dict"]
            else:
                w_star_params = w_star_state
            w_star_vector = parameters_to_vector(
                [param for param in w_star_params.values() if isinstance(param, torch.Tensor)]
            ).detach().cpu()
        else:
            logger.error(f"w_star path {w_star} does not exist. Skipping distance computation.")
            return
    elif isinstance(w_star, nn.Module):
        w_star_vector = parameters_to_vector(list(w_star.parameters())).detach().cpu()
    elif isinstance(w_star, dict):
        w_star_vector = parameters_to_vector(
            [param for param in w_star.values() if isinstance(param, torch.Tensor)]
        ).detach().cpu()
    else:
        logger.error(f"Unsupported type for w_star: {type(w_star)}. Skipping distance computation.")
        return

    distances = []
    for i, model in enumerate(models):
        model_vector = parameters_to_vector(list(model.parameters())).detach().cpu()
        distance = torch.norm(model_vector - w_star_vector).item()
        distances.append(distance)
        logger.info(f"Distance of model {i} to w*: {distance:.4f}")

    mean_distance = np.mean(distances)
    logger.info(f"Mean L2 distance to w*: {mean_distance:.4f}")

    if summary_path:
        try:
            with open(summary_path, "w") as f:
                f.write(f"Mean L2 distance to w*: {mean_distance:.4f}\n")
            logger.info(f"Mean distance saved to {summary_path}")
        except Exception as e:
            logger.error(f"Error saving mean distance to {summary_path}: {e}")



def main():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    args = parse_args()
    config = Config()

    # Defense variant determination
    defense_name_to_use = config.variant
    if args.defense_variant:
        defense_name_to_use = args.defense_variant
        logger.info(f"Overriding defense variant from CLI: {defense_name_to_use}")
    config.variant = defense_name_to_use
    logger.info(f"Running with defense variant: {config.variant}")
    logger.info(f"Attack type from config: {config.attack_type}")

    # Handle analysis-only mode
    if args.analyze_only:
        if not args.result_dir_to_analyze:
            logger.error("--analyze_only requires --result_dir_to_analyze to be set.")
            return
        logger.info(
            f"Analysis only mode. Analyzing results from: {args.result_dir_to_analyze}"
        )
        run_analysis(args.result_dir_to_analyze, config.variant, config.device)
        return

    # Result directory
    dir_suffix = f"_{args.result_dir_suffix}" if args.result_dir_suffix else ""
    config.result_dir = os.path.join(
        "sythetic_results",
        f"results_{config.variant}_{config.attack_type}{dir_suffix}_{config.timestamp}",
    )
    os.makedirs(config.result_dir, exist_ok=True)
    if hasattr(config, "save_config"):
        config.save_config()
    logger.info(f"Results will be saved in {config.result_dir}")

    # Network topology
    adj_matrix_path = os.path.join(config.result_dir, "adjacency_matrix.npy")
    byzantine_path = os.path.join(config.result_dir, "byzantine_indices.npy")

    if os.path.exists(adj_matrix_path) and os.path.exists(byzantine_path) and False:
        adj_matrix = load_adjacency_matrix(adj_matrix_path)
        byzantine_indices = load_byzantine_nodes(byzantine_path)
        logger.info(f"Loaded existing network topology and Byzantine indices.")
    else:
        adj_matrix, graph_obj = create_adjacency_matrix(config)
        byzantine_indices, adj_matrix = select_byzantine_nodes(config, adj_matrix)
        logger.info(f"Created new network topology. Byzantine_indices: {byzantine_indices}")
        np.save(adj_matrix_path, adj_matrix)
        np.save(byzantine_path, np.array(byzantine_indices))

    plot_adjacency_matrix(adj_matrix, graph_obj, byzantine_indices, config.result_dir, config.seed)

    # Data loading
    logger.info("Loading data with train/validation/test split...")
    trainloaders, testloader = load_data_without_validation(config)
    logger.info("Data loaded successfully with proper splits.")

    # Model initialization
    logger.info(f"Initializing {config.num_nodes} models for {config.dataset} on {config.device}...")
    model_name = getattr(config, "model", "cnn")
    model_kwargs = {}
    if config.dataset == "mnist":
        if model_name == "vit":
            ModelClass = MiniViT
            model_kwargs = {"image_size": 28, "in_channels": 1}
        else:
            ModelClass = SimpleCNN
    elif config.dataset == "cifar10":
        if model_name == "vit":
            ModelClass = MiniViT
            model_kwargs = {"image_size": 32, "in_channels": 3}
        else:
            ModelClass = CIFAR10CNN
    elif config.dataset == "synthetic":
        ModelClass = LogisticRegression
    else:
        logger.error(f"Unsupported dataset in main.py: {config.dataset}")
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    w_star_reference: Optional[str] = None
    if config.dataset != "synthetic":
        models = [
            ModelClass(seed=config.seed + i, **model_kwargs).to(config.device)
            for i in range(config.num_nodes)
        ]
    else:
        wstar_path = "wstar.pt"
        w_star_reference = wstar_path
        models = [
            ModelClass(
                seed=config.seed,
                node_id=i,
                wstar_path=wstar_path,
                perturb_mean=config.mean,
                perturb_std=config.std,
                device=config.device,
            ).to(config.device)
            for i in range(config.num_nodes)
        ]
        logger.info(f"Loaded w* from {wstar_path} and initialized models with perturbations.")
    logger.info("Models initialized successfully.")
    init_model_path = os.path.join(config.result_dir, "average_distance.txt")
    save_average_distance(models, config.result_dir, init_model_path, w_star_reference)

    criterion = nn.CrossEntropyLoss()
    if config.dataset == "synthetic":
        criterion = nn.BCEWithLogitsLoss()

    defense_obj: Optional[BaseDefense] = None
    if config.variant in DEFENSE_CLASS_MAP:
        DefenseClass = DEFENSE_CLASS_MAP[config.variant]
        defense_obj = DefenseClass(config=config)
        logger.info(f"Using defense: {defense_obj.__class__.__name__}")
    elif config.variant == "braso":
        logger.info("Using braso method.")
    elif config.variant == "rtc":
        logger.info("Using rtc method.")
    elif config.variant == "ubar":
        logger.info("Using ubar method.")
    elif config.variant == "valid":
        logger.info("Using valid method.")
    elif config.variant == "secure":
        logger.info("Using secureDl method.")
    elif config.variant == "synthetic":
        logger.info("Using synthetic_train method.")
    elif config.variant != "none" and config.variant is not None:
        logger.warning(
            f"Defense variant '{config.variant}' not found. Default aggregation (mean) will be used in train_epoch."
        )
    else:
        logger.info("No explicit defense selected ('none'). Default aggregation (mean) will be used.")

    # --- Metrics ---
    all_epoch_losses_per_node: List[List[float]] = []
    all_epoch_node_accuracies: List[List[float]] = []
    all_mean_train_losses_per_epoch: List[float] = []
    all_mean_test_accuracies_per_epoch: List[float] = []

    # Training loop
    logger.info(f"Starting training for {config.num_epochs} epochs...")
    training_start_time = time.time()
    evaluate_func = evaluate_models
    if config.variant == "braso":
        train_func = braso_epoch
    elif config.variant == "rtc":
        train_func = rtc_epoch
    elif config.variant == "ubar":
        train_func = ubar_epoch
    elif config.variant == "secure":
        train_func = secure_epoch
    elif config.dataset == "synthetic":
        train_func = synthetic_train_epoch
        evaluate_func = synthetic_evaluate_models
    else:
        train_func = train_epoch

    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        current_lr = config.learning_rate
        if hasattr(config, "lr_schedule") and callable(config.lr_schedule):
            current_lr = config.lr_schedule(epoch)
        current_attack_type = config.attack_type

        logger.info(
            f"Epoch {epoch + 1}/{config.num_epochs} | LR: {current_lr:.6f} | Attack: {current_attack_type}"
        )

        mean_loss_this_epoch, losses_per_node_this_epoch = train_func(
            models,
            trainloaders,
            adj_matrix,
            byzantine_indices,
            criterion,
            current_lr,
            defense_obj,
            current_attack_type,
            config,
            epoch,
        )
        all_mean_train_losses_per_epoch.append(mean_loss_this_epoch)
        all_epoch_losses_per_node.append(losses_per_node_this_epoch)

        mean_accuracy_this_epoch, accs_per_node_this_epoch = evaluate_func(
            models, testloader, byzantine_indices, config
        )
        all_mean_test_accuracies_per_epoch.append(mean_accuracy_this_epoch)
        all_epoch_node_accuracies.append(accs_per_node_this_epoch)

        epoch_duration = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch + 1} Summary: MeanTrainLoss={mean_loss_this_epoch:.4f}, "
            f"MeanTestAcc={mean_accuracy_this_epoch:.2f}% (Duration: {epoch_duration:.2f}s)"
        )

    total_training_time = time.time() - training_start_time
    logger.info(f"Training completed in {total_training_time:.2f} seconds.")

    # Save metrics
    logger.info(f"Saving final metrics to {config.result_dir}...")
    torch.save(
        torch.tensor(all_epoch_losses_per_node, dtype=torch.float32),
        os.path.join(config.result_dir, "loss_per_node.pt"),
    )
    torch.save(
        torch.tensor(all_epoch_node_accuracies, dtype=torch.float32),
        os.path.join(config.result_dir, "accuracy_per_node.pt"),
    )
    torch.save(
        torch.tensor(all_mean_train_losses_per_epoch, dtype=torch.float32),
        os.path.join(config.result_dir, "mean_loss.pt"),
    )
    torch.save(
        torch.tensor(all_mean_test_accuracies_per_epoch, dtype=torch.float32),
        os.path.join(config.result_dir, "mean_accuracy.pt"),
    )
    logger.info("All metrics saved successfully.")

    # Final analysis
    logger.info("Running final analysis...")
    if os.path.exists(os.path.join(config.result_dir, "mean_loss.pt")) and os.path.exists(
        os.path.join(config.result_dir, "mean_accuracy.pt")
    ):
        if not os.path.exists(byzantine_path):
            np.save(byzantine_path, np.array(byzantine_indices))
        run_analysis(config.result_dir, config.variant, config.device)

    logger.info(f"Experiment finished. Results are in {config.result_dir}")

    # Save FINAL stats for node0
    final_model_path = os.path.join(config.result_dir, "final_model_node0.pt")
    log_and_save_model_stats(models[0], final_model_path, "Final")


if __name__ == "__main__":
    main()
