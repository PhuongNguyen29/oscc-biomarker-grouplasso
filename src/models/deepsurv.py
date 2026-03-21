"""
DeepSurv Model Implementation
Based on: Katzman et al., 2018, BMC Medical Research Methodology
"DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class DeepSurv(nn.Module):
    """
    DeepSurv: Feed-forward neural network for Cox proportional hazards.
    
    Architecture follows the original paper:
    - Fully connected layers with dropout
    - ReLU/ELU activation functions
    - Single output node for log hazard ratio
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_sizes: List[int] = [512, 256],
        dropout: float = 0.4,
        activation: str = 'relu',
        batch_norm: bool = True,
        weight_init: str = 'kaiming_uniform'
    ):
        """
        Initialize DeepSurv model.
        
        Args:
            n_features: Number of input features (genes)
            hidden_sizes: List of hidden layer sizes (default: [512, 256])
            dropout: Dropout probability (default: 0.4)
            activation: Activation function ('relu', 'elu', 'selu')
            batch_norm: Whether to use batch normalization
            weight_init: Weight initialization method
        """
        super(DeepSurv, self).__init__()
        
        self.n_features = n_features
        self.hidden_sizes = hidden_sizes
        self.dropout_prob = dropout
        self.use_batch_norm = batch_norm
        
        # Select activation function
        self.activation = self._get_activation(activation)
        
        # Build network layers
        layers = []
        prev_size = n_features
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            linear = nn.Linear(prev_size, hidden_size)
            layers.append(('fc{}'.format(i), linear))
            
            # Batch normalization (if enabled)
            if batch_norm:
                layers.append(('bn{}'.format(i), nn.BatchNorm1d(hidden_size)))
            
            # Activation
            layers.append(('act{}'.format(i), self.activation))
            
            # Dropout
            if dropout > 0:
                layers.append(('drop{}'.format(i), nn.Dropout(dropout)))
            
            prev_size = hidden_size
        
        # Output layer (single node for log hazard)
        layers.append(('output', nn.Linear(prev_size, 1)))
        
        # Create sequential model
        self.network = nn.Sequential(OrderedDict(layers))
        
        # Initialize weights
        self._initialize_weights(weight_init)
        
        logger.info(f"DeepSurv initialized: {n_features} → {' → '.join(map(str, hidden_sizes))} → 1")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _get_activation(self, activation: str):
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        return activations.get(activation.lower(), nn.ReLU())
    
    def _initialize_weights(self, method: str):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                elif method == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif method == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input features (batch_size, n_features)
            
        Returns:
            log_hazard: Log hazard ratios (batch_size, 1)
        """
        # Check for NaN/Inf in input - FAIL if found
        if torch.isnan(x).any():
            n_nan = torch.isnan(x).sum().item()
            raise ValueError(f"NaN detected in input features: {n_nan} values. Check data preprocessing.")
        
        if torch.isinf(x).any():
            n_inf = torch.isinf(x).sum().item()
            raise ValueError(f"Inf detected in input features: {n_inf} values. Check data preprocessing.")
        
        # Forward pass
        output = self.network(x)
        
        # Clamp output to prevent overflow in exp() downstream
        output = torch.clamp(output, min=-20, max=20)
        
        # Check for NaN/Inf in output - FAIL if found
        if torch.isnan(output).any():
            n_nan = torch.isnan(output).sum().item()
            raise RuntimeError(
                f"NaN detected in model output: {n_nan} values. "
                f"Likely causes: exploding gradients, extreme dropout ({self.dropout_prob}), "
                f"or tiny layer sizes. Check hyperparameters."
            )
        
        if torch.isinf(output).any():
            n_inf = torch.isinf(output).sum().item()
            raise RuntimeError(
                f"Inf detected in model output: {n_inf} values. "
                f"Likely causes: numerical overflow, extreme learning rate, or unstable architecture."
            )
        
        return output
    
    def predict_risk(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict risk scores (higher = higher risk).
        
        Args:
            x: Input features (batch_size, n_features)
            
        Returns:
            risk_scores: Risk scores (batch_size,)
        """
        with torch.no_grad():
            log_hazard = self.forward(x)
            # Risk score is exp(log_hazard)
            risk = torch.exp(log_hazard).squeeze()
        return risk


class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards Negative Partial Log-Likelihood Loss.
    
    Computes the negative partial log-likelihood using the standard
    sequential cumsum formulation common in neural survival models
    (Katzman et al., 2018; Kvamme et al., 2019). Tied event times
    are processed sequentially rather than grouped, which is an
    approximation when exact ties are present.
    """
    
    def __init__(self):
        """Initialize Cox loss."""
        super(CoxPHLoss, self).__init__()
    
    def forward(
        self,
        log_hazards: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor
    ) -> torch.Tensor:
        """Calculate Cox partial likelihood loss."""
        
        # Validate inputs
        if torch.isnan(log_hazards).any():
            raise ValueError("NaN in log_hazards - model output is invalid")
        if torch.isnan(times).any():
            raise ValueError("NaN in survival times - check data")
        if torch.isnan(events).any():
            raise ValueError("NaN in event indicators - check data")
        
        log_hazards = log_hazards.squeeze()
        
        # Check for events FIRST (before any computation)
        n_events = torch.sum(events)
        if n_events < 1:
            # Return zero loss with gradient (allows backprop but doesn't contribute)
            logger.warning("Batch contains no events - returning zero loss")
            return torch.tensor(0.0, device=log_hazards.device, requires_grad=True)
        
        # Sort by time (descending so cumsum gives correct risk sets)
        sorted_indices = torch.argsort(times, descending=True)
        log_hazards = log_hazards[sorted_indices]
        times = times[sorted_indices]
        events = events[sorted_indices]
        
        # Calculate risk scores
        risk_scores = torch.exp(log_hazards)
        
        # Cumulative sum of risk scores
        # risk_sum[i] = sum of risk_scores[0:i+1] = sum over patients with time >= times[i]
        risk_sum = torch.cumsum(risk_scores, dim=0)
        
        # Log partial likelihood
        log_likelihood = log_hazards - torch.log(risk_sum + 1e-7)
        
        # Check for NaN in loss calculation
        if torch.isnan(log_likelihood).any():
            raise RuntimeError("NaN in log likelihood calculation - numerical instability in Cox loss")
        
        # Only count events
        log_likelihood = log_likelihood * events
        
        # Negative log likelihood (normalized by number of events)
        loss = -torch.sum(log_likelihood) / n_events
        
        if torch.isnan(loss):
            raise RuntimeError("NaN in final loss value - numerical instability")
        
        return loss



class DeepSurvTrainer:
    """
    Training utilities for DeepSurv model.
    """
    
    def __init__(
        self,
        model: DeepSurv,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.0001,
        weight_decay: float = 0.01,
        scheduler_patience: int = 10
    ):
        """
        Initialize trainer.
        
        Args:
            model: DeepSurv model instance
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            scheduler_patience: Patience for learning rate reduction
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        self.criterion = CoxPHLoss()
        
        # Optimizer (Adam as in original paper)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        # Try with verbose first (newer PyTorch), fallback without it
        try:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Maximize C-index
                patience=scheduler_patience,
                factor=0.5,
                verbose=True
            )
        except TypeError:
            # Fallback for older PyTorch versions without verbose parameter
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Maximize C-index
                patience=scheduler_patience,
                factor=0.5
            )
        
        # Training history
        self.history = {
            'train_loss': [],
            'valid_loss': [],
            'train_cindex': [],
            'valid_cindex': []
        }
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch - fail fast on any numerical issues."""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            features = batch['features'].to(self.device)
            times = batch['time'].to(self.device)
            events = batch['event'].to(self.device)
            
            self.optimizer.zero_grad()
            log_hazards = self.model(features)
            loss = self.criterion(log_hazards, times, events)
            loss.backward()
            
            # Check gradient norms
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        raise RuntimeError("NaN in gradients - training is unstable")
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > 100.0:
                # Instead of RuntimeError, return inf to signal failure
                logger.error(f"Exploding gradients: norm={total_norm:.2f}. Marking trial as failed.")
                return float('inf')  # Signal bad trial
            
            if total_norm > 10.0:
                logger.warning(f"Large gradient norm: {total_norm:.2f}")
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def evaluate(self, data_loader) -> Tuple[float, float]:
        """
        Evaluate model on validation/test set.
        
        Args:
            data_loader: Validation/test data loader
            
        Returns:
            avg_loss: Average validation loss
            c_index: Concordance index
        """
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        all_risks = []
        all_times = []
        all_events = []
        
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                times = batch['time'].to(self.device)
                events = batch['event'].to(self.device)
                
                # Forward pass
                log_hazards = self.model(features)
                
                # Calculate loss
                loss = self.criterion(log_hazards, times, events)
                total_loss += loss.item()
                n_batches += 1
                
                # Store predictions for C-index
                risks = torch.exp(log_hazards).squeeze().cpu().numpy()
                all_risks.extend(risks)
                all_times.extend(times.cpu().numpy())
                all_events.extend(events.cpu().numpy())
        
        # Calculate C-index with error handling
        from lifelines.utils import concordance_index
        try:
            c_index = concordance_index(all_times, -np.array(all_risks), all_events)
        except Exception as e:
            logger.error(f"C-index calculation failed: {e}")
            raise RuntimeError(f"Failed to calculate C-index: {e}")
        
        avg_loss = total_loss / n_batches
        return avg_loss, c_index
    
    def fit(
        self,
        train_loader,
        valid_loader,
        n_epochs: int = 100,
        early_stopping_patience: int = 20,
        verbose: bool = True
    ):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            n_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
        """
        best_cindex = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(n_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            valid_loss, valid_cindex = self.evaluate(valid_loader)
            
            # Calculate training C-index
            _, train_cindex = self.evaluate(train_loader)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)
            self.history['train_cindex'].append(train_cindex)
            self.history['valid_cindex'].append(valid_cindex)
            
            # Learning rate scheduling
            self.scheduler.step(valid_cindex)
            
            # Early stopping
            if valid_cindex > best_cindex:
                best_cindex = valid_cindex
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Train Loss: {train_loss:.4f} | C-index: {train_cindex:.4f} | "
                    f"Valid Loss: {valid_loss:.4f} | C-index: {valid_cindex:.4f}"
                )
            
            # Check early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model with C-index: {best_cindex:.4f}")
        
        return self.history


# Utility functions for model evaluation
def calculate_concordance_index(
    model: DeepSurv,
    data_loader,
    device: str = 'cuda'
) -> float:
    """
    Calculate concordance index for model predictions.
    """
    model.eval()
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            risks = model.predict_risk(features).cpu().numpy()
            
            all_risks.extend(risks)
            all_times.extend(batch['time'].numpy())
            all_events.extend(batch['event'].numpy())
    
    from lifelines.utils import concordance_index
    return concordance_index(all_times, -np.array(all_risks), all_events)
