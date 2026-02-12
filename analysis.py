# analysis.py
import torch
import numpy as np
import os
import matplotlib.pyplot as plt


# import seaborn as sns # No longer needed if not plotting distributions or heatmaps
# from models import SimpleCNN # No longer needed if not recreating models

def load_data_for_analysis(result_dir, variant):
    """
    Load saved mean loss and mean accuracy data for analysis.

    Args:
        result_dir (str): Directory containing saved data.
        variant (str): Algorithm variant name (used for logging/titling).

    Returns:
        tuple: (loss_data, accuracy_data)
               loss_data: Tensor of mean losses per epoch.
               accuracy_data: Tensor of mean accuracies per epoch.
    """
    loss_data = None
    accuracy_data = None

    # Load mean loss data
    # Ensure these filenames match what's saved in main.py
    loss_path = os.path.join(result_dir, "mean_loss.pt")
    if os.path.exists(loss_path):
        loss_data = torch.load(loss_path)
        print(f"Loaded mean loss data for {variant} from {loss_path}")
    else:
        print(f"Warning: No mean loss data found at {loss_path}")

    # Load mean accuracy data
    acc_path = os.path.join(result_dir, "mean_accuracy.pt")
    if os.path.exists(acc_path):
        accuracy_data = torch.load(acc_path)
        print(f"Loaded mean accuracy data for {variant} from {acc_path}")
    else:
        print(f"Warning: No mean accuracy data found at {acc_path}")

    # Return only loss and accuracy, models_data is removed
    return loss_data, accuracy_data


def plot_convergence_curves(loss_data, accuracy_data, variant, result_dir):
    """
    Generate convergence plots for mean loss and mean accuracy.

    Args:
        loss_data (torch.Tensor): Tensor of mean loss data per epoch.
        accuracy_data (torch.Tensor): Tensor of mean accuracy data per epoch.
        variant (str): Algorithm variant name (for plot titles/legends).
        result_dir (str): Directory to save plots.
    """
    # Create directory for analysis plots if it doesn't exist
    analysis_plot_dir = os.path.join(result_dir, "convergence_plots")
    os.makedirs(analysis_plot_dir, exist_ok=True)

    if loss_data is None and accuracy_data is None:  # Check if at least one is available
        print("No loss or accuracy data available for plotting convergence curves.")
        return

    num_epochs_loss = len(loss_data) if loss_data is not None else 0
    num_epochs_acc = len(accuracy_data) if accuracy_data is not None else 0

    # Plot mean accuracy over time
    if accuracy_data is not None and num_epochs_acc > 0:
        plt.figure(figsize=(10, 6))
        # Ensure accuracy_data is a 1D tensor of mean accuracies
        if accuracy_data.ndim > 1:  # If it was saved as per-node per-epoch, take mean over nodes
            print("Warning: accuracy_data seems to be per-node. Plotting mean over nodes.")
            mean_acc_values = torch.nanmean(accuracy_data,
                                            dim=1).cpu().numpy() if accuracy_data.ndim > 1 else accuracy_data.cpu().numpy()
        else:
            mean_acc_values = accuracy_data.cpu().numpy()

        epochs_acc = list(range(1, len(mean_acc_values) + 1))
        # mean_acc_values = np.convolve(mean_acc_values,5)
        plt.plot(epochs_acc, mean_acc_values, marker='o', linestyle='-', label=f"{variant} Accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Mean Accuracy (%)')
        plt.title(f'Mean Accuracy Over Time ({variant})')
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(analysis_plot_dir, f"{variant}_mean_accuracy_convergence.png"))
        plt.close()
        print(f"Mean accuracy convergence plot saved to {analysis_plot_dir}")

        # --- Convergence Analysis (optional, based on mean accuracy) ---
        final_acc = mean_acc_values[-1]
        # Threshold for convergence, e.g., 90% of max possible (100%) or 90% of final_acc
        # Let's use 90% of the observed final accuracy as a simple heuristic
        convergence_threshold_acc = 0.9 * final_acc

        convergence_epoch_acc = -1  # Default to not converged
        for epoch_idx, acc_val in enumerate(mean_acc_values):
            if acc_val >= convergence_threshold_acc:
                convergence_epoch_acc = epoch_idx + 1  # epochs are 1-indexed
                break

        last_n_acc = max(1, int(len(mean_acc_values) * 0.1))  # Last 10% of epochs
        stability_acc = np.std(mean_acc_values[-last_n_acc:])

        print(f"Accuracy Analysis for {variant}:")
        print(f"  Final Mean Accuracy: {final_acc:.2f}%")
        if convergence_epoch_acc != -1:
            print(
                f"  Epoch to reach {convergence_threshold_acc * 100:.1f}% accuracy (90% of final): {convergence_epoch_acc}")
        else:
            print(f"  Did not reach 90% of final accuracy within the epochs.")
        print(f"  Accuracy Stability (std dev of last 10% epochs): {stability_acc:.4f}")
        # --- End of Convergence Analysis for Accuracy ---

    else:
        print("Accuracy data not available for plotting.")

    # Plot mean loss over time
    if loss_data is not None and num_epochs_loss > 0:
        plt.figure(figsize=(10, 6))
        # Ensure loss_data is a 1D tensor of mean losses
        if loss_data.ndim > 1:  # If it was saved as per-node per-epoch, take mean over nodes
            print("Warning: loss_data seems to be per-node. Plotting mean over nodes.")
            mean_loss_values = torch.nanmean(loss_data,
                                             dim=1).cpu().numpy() if loss_data.ndim > 1 else loss_data.cpu().numpy()
        else:
            mean_loss_values = loss_data.cpu().numpy()
        # mean_loss_values = np.convolve(mean_loss_values, 5)
        epochs_loss = list(range(1, len(mean_loss_values) + 1))

        plt.plot(epochs_loss, mean_loss_values, marker='x', linestyle='--', color='red', label=f"{variant} Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Mean Loss')
        plt.title(f'Mean Loss Over Time ({variant})')
        plt.grid(True, alpha=0.5)
        plt.yscale('log')  # Log scale is common for loss
        plt.legend()
        plt.savefig(os.path.join(analysis_plot_dir, f"{variant}_mean_loss_convergence.png"))
        plt.close()
        print(f"Mean loss convergence plot saved to {analysis_plot_dir}")
    else:
        print("Loss data not available for plotting.")

    # Save numerical summary (optional, can be expanded)
    summary_file_path = os.path.join(analysis_plot_dir, f"{variant}_convergence_summary.txt")
    with open(summary_file_path, 'w') as f:
        f.write(f"Convergence Analysis Summary for Variant: {variant}\n")
        f.write("=" * 40 + "\n")
        if accuracy_data is not None and num_epochs_acc > 0:
            f.write(f"Final Mean Accuracy: {final_acc:.2f}%\n")
            if convergence_epoch_acc != -1:
                f.write(f"Epoch to reach {convergence_threshold_acc * 100:.1f}% accuracy: {convergence_epoch_acc}\n")
            else:
                f.write(f"Convergence to 90% of final accuracy not met.\n")
            f.write(f"Accuracy Stability (last 10% epochs std dev): {stability_acc:.4f}\n")
        else:
            f.write("Accuracy data not available for summary.\n")
        f.write("-" * 40 + "\n")
        if loss_data is not None and num_epochs_loss > 0:
            f.write(f"Final Mean Loss: {mean_loss_values[-1]:.4f}\n")
            # Add more loss-based metrics if needed
        else:
            f.write("Loss data not available for summary.\n")
    print(f"Convergence summary saved to {summary_file_path}")


def run_analysis(result_dir, variant, device):  # device is no longer used here if not recreating models
    """
    Run analysis focusing on loss and accuracy convergence.

    Args:
        result_dir (str): Directory containing saved data.
        variant (str): Algorithm variant name.
        device (torch.device): Device (not actively used if models aren't loaded).
    """
    print(f"Running simplified analysis (loss & accuracy curves) on data in {result_dir}")

    # Load saved mean loss and mean accuracy data
    # The original load_data_for_analysis also tried to load models_data, which we removed.
    # Adjusting call or function to reflect this.
    loss_data, accuracy_data = load_data_for_analysis(result_dir, variant)
    # load_data_for_analysis now only returns loss_data, accuracy_data

    if loss_data is None and accuracy_data is None:
        print("No data found for analysis (mean_loss.pt or mean_accuracy.pt missing). Make sure the paths are correct.")
        return

    # Removed Byzantine indices loading as it's not used without model analysis
    # Removed model recreation logic (recreate_models, plot_model_similarity_heatmap, plot_parameter_distributions)

    # Plot convergence results for loss and accuracy
    plot_convergence_curves(loss_data, accuracy_data, variant, result_dir)

    print(f"Simplified analysis for {variant} completed.")