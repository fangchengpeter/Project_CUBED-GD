import torch
import numpy as np

def random_attack(params, device, var=0.1):
    """
    Random values attack - add random noise scaled by a factor
    
    Args:
        params (list): List of parameter tensors
        device (torch.device): Device to create tensors on
        var (float): Variance/scale of the random noise
        
    Returns:
        list: Modified parameter tensors
    """
    print(f"Applying random attack with variance {var}")
    return [p + var * torch.randn_like(p).to(device) for p in params]
def constant_attack(params, device, constant=100.0):
    """
    Constant attack - add a constant value to parameters
    
    Args:
        params (list): List of parameter tensors
        device (torch.device): Device to create tensors on
        constant (float): Constant value to add
        
    Returns:
        list: Modified parameter tensors
    """
    return [p + constant * torch.ones_like(p).to(device) for p in params]
def sign_flipping_attack(params, device):
    """
    Sign flipping attack - flips the sign of parameters
    
    Args:
        params (list): List of parameter tensors
        device (torch.device): Device to create tensors on
        
    Returns:
        list: Modified parameter tensors
    """
    return [-1.0 * p for p in params]

def scaled_attack(params, device, scale=10.0):
    """
    Scaled attack - multiply parameters by a large factor
    
    Args:
        params (list): List of parameter tensors
        device (torch.device): Device to create tensors on
        scale (float, optional): Scaling factor. Defaults to 10.0.
        
    Returns:
        list: Modified parameter tensors
    """
    return [scale * p for p in params]


def label_flipping_attack(params, device, source_label=0, target_label=1):
    """
    Label flipping attack - swap the weights for source and target labels
    to cause misclassification
    
    Args:
        params (list): List of parameter tensors
        device (torch.device): Device to create tensors on
        source_label (int): The source label to flip from (default: 0)
        target_label (int): The target label to flip to (default: 1)
        
    Returns:
        list: Modified parameter tensors
    """
    # Create a copy of the parameters to modify
    modified_params = [p.clone() for p in params]
    
    # Assume the last two layers are the classification layer weights and biases
    last_layer_weights = modified_params[-2]  # Weights of the final layer
    last_layer_bias = modified_params[-1]     # Bias of the final layer
    
    # Check if we're working with the expected tensor shapes
    if len(last_layer_weights.shape) == 2:  # For fully connected layers (features x classes)
        num_classes = last_layer_weights.shape[1]
        
        if source_label < num_classes and target_label < num_classes:
            # Simple approach: Swap the weights for source and target labels
            # This directly flips the classification between these two classes
            tmp = last_layer_weights[:, source_label].clone()
            last_layer_weights[:, source_label] = last_layer_weights[:, target_label]
            last_layer_weights[:, target_label] = tmp
            
            # Also swap the bias terms if they exist
            if len(last_layer_bias.shape) == 1:  # Typical bias shape
                tmp_bias = last_layer_bias[source_label].clone()
                last_layer_bias[source_label] = last_layer_bias[target_label]
                last_layer_bias[target_label] = tmp_bias
    
    return modified_params

def get_byzantine_params(original_params, attack_type, device, config=None, honest_params=None):
    """
    Generate Byzantine parameters based on attack type
    
    Args:
        original_params (list): Original parameter tensors
        attack_type (str): Type of attack ("random", "sign_flipping", "scaled", "label_flipping", 
                           "targeted", "backdoor", "trimmed_mean")
        device (torch.device): Device to create tensors on
        config (Config, optional): Configuration object for additional parameters
        honest_params (list): List of honest nodes' parameter sets for certain attack types
        
    Returns:
        list: Modified parameter tensors for Byzantine attack
    """
    try:
        random_attack_var = 0.01  # Default value
        if config and hasattr(config, 'random_attack_var'):
            random_attack_var = config.random_attack_var
            
        if attack_type == "random":
            return random_attack(original_params, device, var=random_attack_var)
        elif attack_type == "sign_flipping":
            return sign_flipping_attack(original_params, device)
        elif attack_type == "scaled":
            return scaled_attack(original_params, device)
        elif attack_type == "label_flipping":
            # Parse configuration parameters if available
            source_label = 0
            target_label = 1
            
            if config:
                if hasattr(config, 'source_label'):
                    source_label = config.source_label
                if hasattr(config, 'target_label'):
                    target_label = config.target_label
                    
            return label_flipping_attack(original_params, device, source_label, target_label)
        elif attack_type == "constant":
            constant_value = 100.0
            if config and hasattr(config, 'constant_attack_value'):
                constant_value = config.constant_attack_value
            return constant_attack(original_params, device, constant=constant_value)
        else:
            print(f"Warning: Unknown attack type '{attack_type}'. Using original parameters.")
            return original_params  # Default case - no attack
    except Exception as e:
        print(f"Error in Byzantine attack: {str(e)}")
        return original_params  # Return original params in case of error