import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import torch

def plot_results(all_epoch_losses, all_epoch_accuracies, byzantine_indices, variant, result_dir):
    """
    Plot training losses and test accuracies
    
    Args:
        all_epoch_losses (torch.Tensor): Tensor of losses
        all_epoch_accuracies (torch.Tensor): Tensor of accuracies
        byzantine_indices (list): Indices of Byzantine nodes
        variant (str): Algorithm variant being used
        result_dir (str): Directory to save plots
    """
    # Create figure for accuracy and loss plots
    plt.figure(figsize=(12, 10))
    
    # Plot accuracy curve
    plt.subplot(2, 1, 1)
    
    if all_epoch_accuracies.shape[0] > 0:
        # Get data from tensor
        num_epochs = all_epoch_accuracies.shape[0]
        epochs = list(range(1, num_epochs + 1))
        
        # Plot mean accuracy
        plt.plot(epochs, all_epoch_accuracies, label=f"{variant}", color='blue', linewidth=2)
    else:
        plt.text(0.5, 0.5, "No accuracy data available", ha='center', va='center', transform=plt.gca().transAxes)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy ({variant})")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)  # Accuracy ranges from 0 to 100%
    plt.legend(loc='lower right')

    # Plot loss curve
    plt.subplot(2, 1, 2)
    
    if all_epoch_losses.shape[0] > 0:
        # Get data from tensor
        num_epochs = all_epoch_losses.shape[0]
        epochs = list(range(1, num_epochs + 1))
        
        # Plot mean loss
        plt.plot(epochs, all_epoch_losses, label=f"{variant}", color='red', linewidth=2)
    else:
        plt.text(0.5, 0.5, "No loss data available", ha='center', va='center', transform=plt.gca().transAxes)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss ({variant})")
    plt.grid(True, alpha=0.3)
    
    # Use log scale for loss
    if all_epoch_losses.shape[0] > 0:
        plt.yscale("log")
    
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "accuracy_loss.png"), dpi=300)
    print(f"Plots saved to {result_dir}")


def plot_adjacency_matrix(adj_matrix, graph, byzantine_indices, result_dir, seed):
    """
    Visualize network topology with Byzantine nodes highlighted

    Args:
        adj_matrix (numpy.ndarray): Adjacency matrix
        graph (networkx.Graph): NetworkX graph representation
        byzantine_indices (list): Indices of Byzantine nodes
        result_dir (str): Directory to save plots
        seed (int): Random seed for layout
    """
    # Create a separate figure for network topology
    plt.figure(figsize=(10, 10))

    # Convert byzantine_indices to a Python list if it's not already
    if not isinstance(byzantine_indices, list):
        byzantine_indices = byzantine_indices.tolist() if hasattr(byzantine_indices, 'tolist') else list(
            byzantine_indices)

    # Use spring layout with fixed seed for reproducibility
    pos = nx.spring_layout(graph, seed=seed)

    # Draw normal nodes
    non_byz_nodes = [i for i in range(adj_matrix.shape[0]) if i not in byzantine_indices]
    nx.draw_networkx_nodes(graph, pos, nodelist=non_byz_nodes, node_color='blue',
                           node_size=300, alpha=0.8, label='Honest')

    # Draw Byzantine nodes
    if byzantine_indices:
        nx.draw_networkx_nodes(graph, pos, nodelist=byzantine_indices, node_color='red',
                               node_size=300, alpha=0.8, label='Byzantine')

    # Draw network connections
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)

    # Add node labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family='sans-serif')

    plt.title(
        f"Network Topology (Red: Byzantine, Blue: Honest)\n{len(non_byz_nodes)} Honest Nodes, {len(byzantine_indices)} Byzantine Nodes")
    plt.legend()
    plt.axis('off')

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "network_topology.png"), dpi=300)

    # Create a separate heatmap of the adjacency matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(adj_matrix, cmap='Blues', interpolation='none')
    plt.colorbar(label='Connection')

    # Highlight Byzantine nodes
    if byzantine_indices:
        for idx in byzantine_indices:
            plt.axhline(y=idx, color='red', alpha=0.3)
            plt.axvline(x=idx, color='red', alpha=0.3)

    plt.title("Adjacency Matrix")
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.savefig(os.path.join(result_dir, "adjacency_matrix.png"), dpi=300)
def plot_model_variance(models, variant, byzantine_indices, config, epoch, result_dir):
    """
    Plot model parameter variance across honest nodes
    
    Args:
        models (list): List of models for each node
        variant (str): Algorithm variant being used
        byzantine_indices (list): List of byzantine node indices
        config (Config): Configuration object
        epoch (int): Current epoch
        result_dir (str): Directory to save plot
        
    Returns:
        float: Variance value
    """
    # Create directory for variance plots
    variance_dir = os.path.join(result_dir, "variance")
    os.makedirs(variance_dir, exist_ok=True)
    
    variance = 0.0
    
    # Get honest node indices
    honest_indices = [i for i in range(config.num_nodes) if i not in byzantine_indices]
    
    if len(honest_indices) <= 1:
        print(f"Warning: Not enough honest nodes to compute variance")
        return 0.0
        
    # Collect parameters from honest nodes
    params_list = []
    for node_idx in honest_indices:
        params = [param.data.clone().view(-1) for param in models[node_idx].parameters()]
        params_concatenated = torch.cat(params)
        params_list.append(params_concatenated)
        
    # Stack parameters
    params_tensor = torch.stack(params_list)
    
    # Compute variance across nodes for each parameter
    variance = torch.var(params_tensor, dim=0).mean().item()
    
    # Plot variance over time (save value to a CSV for later plotting)
    variance_file = os.path.join(variance_dir, "variance_over_time.csv")
    if not os.path.exists(variance_file):
        with open(variance_file, 'w') as f:
            f.write("epoch,variance\n")
    
    with open(variance_file, 'a') as f:
        f.write(f"{epoch},{variance}\n")
    
    # If we have accumulated some data points, plot variance over time
    if epoch % 10 == 0 or epoch == config.num_epochs - 1:
        try:
            # Read the CSV file
            epochs = []
            variances = []
            with open(variance_file, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    e, v = line.strip().split(',')
                    epochs.append(int(e))
                    variances.append(float(v))
            
            # Plot variance over time
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, variances, marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Parameter Variance')
            plt.title(f'Model Parameter Variance Over Time ({variant})')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Log scale is often better for variance
            plt.savefig(os.path.join(variance_dir, "variance_over_time.png"))
            plt.close()
        except Exception as e:
            print(f"Error plotting variance over time: {str(e)}")
    
    return variance