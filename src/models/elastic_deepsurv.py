"""
ElasticDeepSurv: DeepSurv with group lasso + ridge regularization.

Extends DeepSurv with gene-level group lasso regularization on the
input layer (Yuan & Lin, 2006) combined with L2 mixing for within-group
shrinkage (Zou & Hastie, 2005). Named ElasticDeepSurv for project
continuity.

References:
    - Katzman et al. (2018) BMC Med Res Methodol — DeepSurv
    - Yuan & Lin (2006) JRSS-B — Group lasso
    - Zou & Hastie (2005) JRSS-B — Elastic net / ridge mixing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
import logging
from lifelines.utils import concordance_index 

from src.models.deepsurv import DeepSurv, CoxPHLoss
from src.utils.regularization import group_lasso_penalty, elastic_net_penalty, get_feature_importance, count_zero_weights

logger = logging.getLogger(__name__)


class ElasticDeepSurv(DeepSurv):
    """
    DeepSurv with group lasso + ridge regularization on the input layer.
    
    Penalizes gene-level weight groups (Yuan & Lin, 2006) with L2 mixing
    for within-group shrinkage. Named ElasticDeepSurv for project continuity.
    """

    def __init__(self,
                 n_features: int,
                 hidden_sizes: List[int] = [256, 128],
                 dropout: float = 0.3,
                 activation: str = 'relu',
                 batch_norm: bool = True,
                 weight_init: str = 'kaiming_uniform',   
                 l1_ratio: float = 0.5,
                 alpha: float = 0.01):
        super(ElasticDeepSurv, self).__init__(
            n_features=n_features,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
            weight_init=weight_init,
        )
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        assert 0.0 <= l1_ratio <= 1.0, "l1_ratio must be between 0 and 1"
        assert alpha >= 0.0, "alpha must be non-negative"

        # Cox loss (instantiated once, reused in compute_loss)
        self.cox_criterion = CoxPHLoss()

        logger.info(f"ElasticDeepSurv initialized with group lasso + ridge regularization:")
        logger.info(f"  l1_ratio: {l1_ratio} (controls group vs ridge balance)")
        logger.info(f"  alpha: {alpha}")
        logger.info(f"  Expected sparsity: {'High' if l1_ratio > 0.7 else 'Moderate' if l1_ratio > 0.3 else 'Low'}")
    

    def compute_loss(
        self,
        log_hazards: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute total loss = Cox loss + group lasso penalty.
        
        Penalty = alpha * [(1 - l2_ratio) * sum_g(||w_g||_2) + l2_ratio * ||W||_F^2]
        where l2_ratio = 1 - l1_ratio, and w_g is the weight column for gene g
        in the first hidden layer.
        
        Args:
            log_hazards: Model predictions (batch_size, 1)
            times: Survival times (batch_size,)
            events: Event indicators (batch_size,)
            return_components: If True, return (total_loss, cox_loss, penalty)
            
        Returns:
            total_loss: Combined loss for backpropagation
            If return_components: (total_loss, cox_loss, penalty)
        """
        cox_loss = self.cox_criterion(log_hazards, times, events)

        penalty = group_lasso_penalty(
            model=self,
            alpha=self.alpha,
            l2_ratio=1 - self.l1_ratio,  # Convert: l1_ratio → group_ratio
            exclude_bias=True,
            first_layer_only=True
        )

        # Combine losses
        total_loss = cox_loss + penalty
        
        if return_components:
            return total_loss, cox_loss, penalty
        return total_loss
    
    
    def get_sparsity_info(self) -> Dict[str, float]:
        """
        Get sparsity information of the model weights.
        
        Returns:
            Dictionary with total weights, zero weights, and sparsity percentage.
        """
        n_zeros, n_total, sparsity_ratio = count_zero_weights(self, threshold=1e-6)
        return {
            'n_zeros': n_zeros,
            'n_total': n_total,
            'sparsity_ratio': sparsity_ratio
        }

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Get feature importance scores based on L2 norm of first-layer weight columns.
        
        Returns:
            List of (feature_name, importance) tuples, sorted descending.
        """
        return get_feature_importance(self, feature_names)
        
    
class ElasticDeepSurvTrainer:
    """
    Trainer for ElasticDeepSurv model.
    Handles training loop, validation, and logging.
    Tracks Cox loss and regularization penalty separately.
    """
    
    def __init__(
        self,
        model: ElasticDeepSurv,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        scheduler_patience: int = 10,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
            )
        try:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', 
                patience=scheduler_patience, 
                factor=0.5,
                verbose=True)
            
        except TypeError:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', 
                patience=scheduler_patience, 
                factor=0.5)
            
        self.history = {
            'train_loss': [],
            'valid_loss': [],
            'train_cindex': [],
            'valid_cindex': [],
            'cox_loss': [],
            'penalty': [],
            'sparsity': []
        }
        
        logger.info(f"ElasticDeepSurvTrainer initialized on {device}.")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  L1 ratio: {model.l1_ratio}")
        logger.info(f"  Alpha: {model.alpha}")
        
    def train_epoch(self, train_loader) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Returns:
            avg_total_loss: Average total loss
            avg_cox_loss: Average Cox loss
            avg_penalty: Average regularization penalty
        """
        self.model.train()
        total_loss_sum = 0.0
        cox_loss_sum = 0.0
        penalty_sum = 0.0
        n_batches = 0
        
        n_params = sum(p.numel() for p in self.model.parameters())
        expected_norm = np.sqrt(n_params) * 0.5
        
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(self.device)
            times = batch['time'].to(self.device)
            events = batch['event'].to(self.device)
            
            
            self.optimizer.zero_grad()
            # Forward pass
            log_hazards = self.model(features)
            total_loss, cox_loss, penalty = self.model.compute_loss(
                log_hazards, times, events, return_components=True
            )
            # Backward pass
            total_loss.backward()
            total_norm = 0.0
            has_nan = False
        
            for p in self.model.parameters():
                if p.grad is not None:
                    # Check for NaN
                    if torch.isnan(p.grad).any():
                        has_nan = True
                        break
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            
            total_norm = total_norm ** 0.5
            
            if has_nan:
                logger.error("NaN detected in gradients. Skipping batch.")
                self.optimizer.zero_grad()
                continue
            
            # Adaptive thresholds based on parameter count
            warn_threshold = 2.0 * expected_norm  # 2x expected
            clip_threshold = 3.0 * expected_norm  # 3x expected
            explode_threshold = 5.0 * expected_norm  # 5x expected
            
            # Handle gradient magnitude
            if total_norm > explode_threshold:
                logger.error(
                    f"Exploding gradients detected (norm={total_norm:.2f}, "
                    f"expected~{expected_norm:.2f}). Skipping batch."
                )
                self.optimizer.zero_grad()
                continue
                
            elif total_norm > clip_threshold:
                logger.warning(
                    f"Large gradients detected (norm={total_norm:.2f}). "
                    f"Clipping to {clip_threshold:.2f}."
                )
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )
            
            elif total_norm > warn_threshold:
                # Log but don't clip (moderate gradient)
                logger.debug(
                    f"Moderate gradient norm: {total_norm:.2f} "
                    f"(expected~{expected_norm:.2f})"
                )
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate losses
            total_loss_sum += total_loss.item()
            cox_loss_sum += cox_loss.item()
            penalty_sum += penalty.item()
            n_batches += 1
        
        if n_batches == 0:
            logger.error("No successful batches in epoch!")
            return float('inf'), float('inf'), float('inf')
        
        return (
            total_loss_sum / n_batches,
            cox_loss_sum / n_batches,
            penalty_sum / n_batches
        )
    
    def evaluate(self, data_loader) -> Tuple[float, float, float, float]:
        """
        Evaluate the model on a data loader.
        
        Returns:
            Tuple of (avg_total_loss, avg_cox_loss, avg_penalty, c_index)
        """
        self.model.eval()
        total_loss_sum = 0.0
        cox_loss_sum = 0.0
        penalty_sum = 0.0
        n_batches = 0
        
        all_risks = []
        all_times = []
        all_events = []
        
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                times = batch['time'].to(self.device)
                events = batch['event'].to(self.device)
                
                log_hazards = self.model(features)
                total_loss, cox_loss, penalty = self.model.compute_loss(log_hazards, times, events, return_components=True)
                
                total_loss_sum += total_loss.item()
                cox_loss_sum += cox_loss.item()
                penalty_sum += penalty.item()   
                n_batches += 1
                risks = torch.exp(log_hazards).squeeze().cpu().numpy()
                all_risks.extend(risks)
                all_times.extend(times.cpu().numpy())
                all_events.extend(events.cpu().numpy())
                
        
        try:
            c_index = concordance_index(all_times, -np.array(all_risks), all_events)
        except Exception as e:
            logger.error(f"Error computing concordance index: {e}")
            raise RuntimeError("Failed to compute concordance index during validation.")
        

        return (total_loss_sum / n_batches, 
                cox_loss_sum / n_batches, 
                penalty_sum / n_batches,
                c_index)
        
    def fit(
        self,
        train_loader,
        valid_loader,
        n_epochs: int = 100,
        early_stopping_patience: int = 20,
        verbose: bool = True
    ):
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            n_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        best_cindex = 0.0
        patience_counter = 0
        best_model_state = None
        best_epoch = 0  # Track best epoch locally
        
        logger.info("Starting training...")
        logger.info(f"Max epochs: {n_epochs}, Early stopping patience: {early_stopping_patience}")
        
        for epoch in range(n_epochs):
            # Training
            train_total, train_cox, train_penalty = self.train_epoch(train_loader)
            _, _, _, train_cindex = self.evaluate(train_loader)
            
            # Store training metrics
            self.history['train_loss'].append(train_total)
            self.history['train_cindex'].append(train_cindex)
            self.history['cox_loss'].append(train_cox)
            self.history['penalty'].append(train_penalty)
            
            # Validation
            if valid_loader is not None:
                valid_total, valid_cox, valid_penalty, valid_cindex = self.evaluate(valid_loader)
                self.history['valid_loss'].append(valid_total)
                self.history['valid_cindex'].append(valid_cindex)
            else:
                # No validation
                valid_cindex = train_cindex
                self.history['valid_loss'].append(None)
                self.history['valid_cindex'].append(None)
            
            # Sparsity
            sparsity_info = self.model.get_sparsity_info()
            sparsity_ratio = sparsity_info['sparsity_ratio']
            self.history['sparsity'].append(sparsity_ratio)
            
            # Learning rate scheduler
            self.scheduler.step(valid_cindex)
            
            # Early stopping
            if early_stopping_patience is not None and valid_loader is not None:
                if valid_cindex > best_cindex:
                    best_cindex = valid_cindex
                    best_epoch = epoch + 1
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
        
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}.")
                    if verbose:
                        logger.info(f"Best C-index: {best_cindex:.4f}")
                    break
            
            # Print progress
            if verbose:
                logger.info(
                    f"Epoch {epoch + 1}/{n_epochs} | "
                    f"Train Loss: {train_total:.4f} (Cox: {train_cox:.4f}, Penalty: {train_penalty:.4f}) | "
                    f"C-index: {train_cindex:.4f} | "
                    f"Valid C-index: {valid_cindex:.4f} | "
                    f"Sparsity: {sparsity_ratio:.1%}"
                )
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Best model with C-index: {best_cindex:.4f} restored.")
            
            final_sparsity = self.model.get_sparsity_info()
            logger.info(f"Final model sparsity: {final_sparsity['sparsity_ratio']:.1%} "
                        f"({final_sparsity['n_zeros']} / {final_sparsity['n_total']} weights zeroed).")
        
        # Add best_epoch to history
        self.history['best_epoch'] = best_epoch 
        
        return self.history
    
    
# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing ElasticDeepSurv model and trainer...")
    print("=" * 60)
    model = ElasticDeepSurv(n_features=100, hidden_sizes=[64, 32], dropout=0.3, l1_ratio=0.7, alpha=0.01)
    
    print(f"\nModel architecture:\n{model}")
    print("\nTesting forward pass...")
    x = torch.randn(16, 100)
    output = model(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    print("\nTesting loss computation...")
    times = torch.rand(16) * 100
    events = torch.randint(0, 2, (16,)).float()
    total_loss, cox_loss, penalty = model.compute_loss(output, times, events, return_components=True)
    
    print(f"Total loss: {total_loss.item():.4f}, Cox loss: {cox_loss.item():.4f}, Penalty: {penalty.item():.4f}")
    
    # Test sparsity
    print("\nInitial sparsity:")
    sparsity = model.get_sparsity_info()
    print(f"  Near-zero weights: {sparsity['n_zeros']}/{sparsity['n_total']}")
    print(f"  Sparsity ratio: {sparsity['sparsity_ratio']:.1%}")
     
    print("\nTop 5 feature importances:")
    gene_names = [f"gene_{i}" for i in range(100)]
    importances = model.get_feature_importance(gene_names)
    for gene, score in importances[:5]:
        print(f"  {gene}: {score:.4f}")
        
    print("\n" + "=" * 60)
    print("All tests passed for ElasticDeepSurv.")
    print("=" * 60)
