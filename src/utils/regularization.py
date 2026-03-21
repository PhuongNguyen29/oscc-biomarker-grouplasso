"""
Regularization Utilities for Neural Networks

Provides penalty functions used in the ElasticDeepSurv pipeline:

- group_lasso_penalty: Group lasso with L2 mixing on the input layer.
  Encourages gene-level sparsity by penalizing the L2 norm of each
  input feature's weight group (Yuan & Lin, 2006), combined with
  Frobenius-norm shrinkage for within-group stability (Zou & Hastie, 2005).
  This is the active penalty used in ElasticDeepSurv.

- elastic_net_penalty: Standard elastic net (Zou & Hastie, 2005).
  Retained for reference but not used in the current pipeline.

- l1_penalty, l2_penalty: Component penalties.

References:
- Yuan & Lin (2006). "Model selection and estimation in regression
  with grouped variables." JRSS-B, 68(1), 49-67.
- Zou & Hastie (2005). "Regularization and variable selection via the
  elastic net." JRSS-B, 67(2), 301-320.
- Friedman et al. (2010). "Regularization paths for generalized linear
  models via coordinate descent." JSS, 33(1), 1-22.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import numpy as np


def l1_penalty(model: nn.Module, exclude_bias: bool = True) -> torch.Tensor:
    """Compute L1 penalty (sum of absolute values) for model weights.

    Args:
        model (nn.Module): Neural network model.
        exclude_bias (bool): Whether to exclude bias terms from penalty.

    Returns:
        torch.Tensor: Computed L1 penalty.
    """
    l1_norm = 0.0
    for name, param in model.named_parameters():
        if exclude_bias and 'bias' in name:
            continue
        l1_norm += torch.sum(torch.abs(param))
    return l1_norm  

def group_lasso_penalty(
    model: nn.Module,
    alpha: float,
    l2_ratio: float = 0.3,
    exclude_bias: bool = True,
    first_layer_only: bool = True
) -> torch.Tensor:
    """
    Group lasso with L2 mixing for gene-level sparsity.
    
    Groups all connections from each input feature (gene) together and
    penalizes the L2 norm of each group, encouraging entire genes to be
    zeroed out. Combined with Frobenius-norm (ridge) shrinkage for
    within-group stability.
    
    Penalty = alpha * [(1 - l2_ratio) * sum_g(||w_g||_2) + l2_ratio * ||W||_F^2]
    
    where w_g is the column of the weight matrix corresponding to gene g.
    
    Based on:
    - Yuan & Lin (2006). "Model selection and estimation in regression 
      with grouped variables." JRSS-B. (group lasso component)
    - Zou & Hastie (2005). "Regularization and variable selection via
      the elastic net." JRSS-B. (L2/ridge mixing for stability)
    
    Note: This is NOT the sparse group lasso of Simon et al. (2013),
    which uses elementwise L1 within groups. Our within-group penalty
    is L2 (Frobenius norm squared), providing shrinkage without
    within-group sparsity.
    
    Args:
        model: Neural network model
        alpha: Overall regularization strength
        l2_ratio: Mix with L2 for stability (0=pure group lasso, 1=pure L2)
        exclude_bias: Whether to exclude bias terms
        first_layer_only: Apply only to first layer (input->hidden)
        
    Returns:
        Group lasso + ridge penalty term
        
    Note:
        For a weight matrix W of shape [hidden_dim, input_dim]:
        - Each column represents one input feature (gene)
        - Group lasso penalty = sum over genes(||column_i||_2)
        - This forces entire genes to zero, not just individual weights
    """
    penalty = 0.0
    layer_count = 0
    
    for name, param in model.named_parameters():
        # Skip bias if requested
        if exclude_bias and 'bias' in name:
            continue
            
        # Only apply to weight matrices
        if 'weight' not in name:
            continue
            
        # If first_layer_only, skip after first layer
        if first_layer_only and layer_count > 0:
            break
            
        if 'weight' in name:
            # param shape: [output_dim, input_dim] e.g., [256, 308]
            # We want to group by input features (genes) = columns
            
            # Compute L2 norm for each input feature's group
            # dim=0 means we sum over output dimension (hidden units)
            gene_group_norms = torch.norm(param, p=2, dim=0)  # Shape: [input_dim]
            
            # Group Lasso penalty: sum of group norms
            group_penalty = torch.sum(gene_group_norms)
            
            # L2 penalty for stability (Frobenius norm squared)
            l2_penalty = torch.sum(param ** 2)
            
            # Combined penalty
            penalty += alpha * ((1 - l2_ratio) * group_penalty + l2_ratio * l2_penalty)
            
            layer_count += 1
    
    return penalty

def l2_penalty(model: nn.Module, exclude_bias: bool = True) -> torch.Tensor:
    """Compute L2 penalty (squared Frobenius norm) for model weights.

    Args:
        model (nn.Module): Neural network model.
        exclude_bias (bool): Whether to exclude bias terms from penalty.

    Returns:
        torch.Tensor: Computed L2 penalty.
    """
    l2_norm = 0.0
    for name, param in model.named_parameters():
        if exclude_bias and 'bias' in name:
            continue
        l2_norm += torch.sum(param ** 2)
    return l2_norm

def elastic_net_penalty(
    model: nn.Module, 
    alpha: float, 
    l1_ratio: float, 
    exclude_bias: bool = True
) -> torch.Tensor:
    """Compute Elastic Net penalty for model parameters.
    
    Note: This function is retained for reference but is NOT used in
    the current ElasticDeepSurv pipeline. The active penalty is
    group_lasso_penalty.

    Args:
        model (nn.Module): Neural network model.
        alpha (float): Overall regularization strength.
        l1_ratio (float): Balance between L1 and L2 penalties.
        exclude_bias (bool): Whether to exclude bias terms from penalty.

    Returns:
        torch.Tensor: Computed Elastic Net penalty.
    """
    assert 0.0 <= l1_ratio <= 1.0, "l1_ratio must be between 0 and 1"
    assert alpha >= 0.0, "alpha must be non-negative"
    l1_norm = l1_penalty(model, exclude_bias)
    l2_norm = l2_penalty(model, exclude_bias)
    elastic_penalty = alpha * (l1_ratio * l1_norm + (1 - l1_ratio) * l2_norm)
    return elastic_penalty

def get_feature_importance(
    model: nn.Module, 
    feature_names: List[str], 
    top_k: Optional[int] = None
) -> List[Tuple[str, float]]:
    """Get feature importance based on L2 norm of first-layer weight columns.

    Args:
        model (nn.Module): Neural network model.
        feature_names (List[str]): List of feature names corresponding to input layer.
        top_k (Optional[int]): Number of top features to return. If None, return all.

    Returns:
        List[Tuple[str, float]]: List of tuples (feature_name, importance),
            sorted by importance descending.
    """
    # Get first layer weights
    first_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            first_layer = module
            break
    if first_layer is None:
        raise ValueError("Model does not contain a Linear layer.")
    
    weights = first_layer.weight.data.cpu().numpy()  # shape (output_dim, input_dim)
    # Compute importance as L2 norm across hidden units for each feature
    importance_scores = np.linalg.norm(weights, axis=0)
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importance_scores))]
        
    feature_importance = list(zip(feature_names, importance_scores))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    if top_k is not None:
        return feature_importance[:top_k]
    
    return feature_importance

def count_zero_weights(model: nn.Module, threshold: float = 1e-6) -> Tuple[int, int, float]:
    """Count number of weights that are effectively zero.

    Args:
        model (nn.Module): Neural network model.
        threshold (float): Threshold below which weights are considered zero.

    Returns:
        Tuple of (n_zero, n_total, sparsity_ratio).
    """
    zero_count = 0
    n_total = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            param_np = param.detach().cpu().numpy()
            zero_count += np.sum(np.abs(param_np) < threshold)
            n_total += param_np.size
    return zero_count, n_total, zero_count / n_total if n_total > 0 else 0.0    

def get_regularization_path(
    losses: List[float],
    penalties: List[float],
    alphas: List[float]
) -> dict:
    """Get regularization path data.

    Args:
        losses (List[float]): List of training losses.
        penalties (List[float]): List of regularization penalties.
        alphas (List[float]): List of alpha values used.

    Returns:
        dict: Dictionary containing losses, penalties, and alphas.
    """
    return {
        'losses': losses,
        'penalties': penalties,
        'alphas': alphas,
        'total_losses': [l + p for l, p in zip(losses, penalties)],
        'optimal_idx': np.argmin([l + p for l, p in zip(losses, penalties)]),
        'optimal_alpha': alphas[np.argmin([l + p for l, p in zip(losses, penalties)])]
    }

if __name__ == "__main__":
    import torch.nn as nn
    print("=" * 60)
    print("Testing Regularization Utilities")
    print("=" * 60)

    # Example usage
    model = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    l1 = l1_penalty(model)
    print(f"L1 Penalty: {l1.item():.4f}")
    l2 = l2_penalty(model)
    print(f"L2 Penalty: {l2.item():.4f}")
    
    # Test Elastic Net with different ratios
    print("\nElastic Net Penalties with different l1_ratios:")
    for l1_ratio in [0.0, 0.3, 0.5, 0.7, 1.0]:
        penalty = elastic_net_penalty(model, alpha=0.1, l1_ratio=l1_ratio)
        print(f"  L1 Ratio: {l1_ratio:.1f}, Elastic Net Penalty: {penalty.item():.4f}")

    # Test Group Lasso
    print("\nGroup Lasso Penalty (first layer only):")
    for l2_ratio in [0.0, 0.3, 0.5, 0.7, 1.0]:
        penalty = group_lasso_penalty(model, alpha=0.1, l2_ratio=l2_ratio)
        print(f"  L2 Ratio: {l2_ratio:.1f}, Group Lasso Penalty: {penalty.item():.4f}")

    zeros, total, ratio = count_zero_weights(model, threshold=1e-6)
    print(f"\nSparsity: {zeros}/{total} weights are effectively zero ({ratio*100:.2f}%)")
    
    # Test feature importance
    gene_names = [f"GENE_{i}" for i in range(100)]
    importances = get_feature_importance(model, gene_names)
    print("\nTop 5 most important features (all):")
    for name, score in importances[:5]:
        print(f"  {name}: {score:.4f}")

    # Test top_k
    top3 = get_feature_importance(model, gene_names, top_k=3)
    print(f"\nTop 3 via top_k parameter: {len(top3)} features returned")
        
    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)
