import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class SurvivalDataset(Dataset):
    """
    PyTorch Dataset for survival analysis with gene expression data.
    Based on DeepSurv (Katzman et al., 2018) and PyCox framework.
    
    Handles censored survival data with (time, event) outcomes.
    """
    
    def __init__(self, 
                 expression_df: pd.DataFrame,
                 survival_df: pd.DataFrame,
                 transform=None):
        """
        Args:
            expression_df: Gene expression matrix (genes × samples)
            survival_df: Survival data with 'time' and 'event' columns
            transform: Optional transform to apply to features
        """
        # Transpose expression to (samples × genes) for ML
        self.features = expression_df.T
        
        # Align survival data with expression samples
        common_samples = list(set(self.features.index) & set(survival_df.index))
        if len(common_samples) < len(self.features):
            logger.warning(f"Only {len(common_samples)}/{len(self.features)} samples have survival data")
        
        # Sort to ensure alignment
        common_samples = sorted(common_samples)
        self.features = self.features.loc[common_samples]
        self.survival = survival_df.loc[common_samples]
        
        # Convert to numpy arrays
        self.X = self.features.values.astype(np.float32)
        self.y_time = self.survival['time'].values.astype(np.float32)
        self.y_event = self.survival['event'].values.astype(np.float32)
        
        # Store metadata
        self.gene_names = expression_df.index.tolist()
        self.sample_ids = common_samples
        self.n_features = self.X.shape[1]
        
        self.transform = transform
        
        logger.info(f"Dataset created: {len(self)} samples, {self.n_features} features")
        logger.info(f"Event rate: {self.y_event.mean():.2%}")
        logger.info(f"Median survival time: {np.median(self.y_time):.2f}")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary with:
        - 'features': gene expression vector
        - 'time': survival time
        - 'event': event indicator (1=event, 0=censored)
        """
        features = self.X[idx]
        
        if self.transform:
            features = self.transform(features)
            
        return {
            'features': torch.from_numpy(features),
            'time': torch.tensor(self.y_time[idx]),
            'event': torch.tensor(self.y_event[idx]),
            'sample_id': self.sample_ids[idx]
        }
    
    def get_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return full dataset as numpy arrays (for non-batch training)"""
        return self.X, self.y_time, self.y_event
    
    def create_train_valid_split(self, valid_size: float = 0.2, random_seed: int = 42):
        """
        Create train/validation split stratified by event status.
        
        Based on "Effective ways of splitting data for survival analysis" 
        (Royston & Altman, 2013, Statistics in Medicine)
        """
        from sklearn.model_selection import train_test_split
        
        # Stratify by event status to maintain proportion
        indices = np.arange(len(self))
        train_idx, valid_idx = train_test_split(
            indices, 
            test_size=valid_size,
            stratify=self.y_event,
            random_state=random_seed
        )
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(self, train_idx)
        valid_dataset = torch.utils.data.Subset(self, valid_idx)
        
        logger.info(f"Split: Train={len(train_idx)}, Valid={len(valid_idx)}")
        
        return train_dataset, valid_dataset


class CombinedSurvivalDataset(Dataset):
    """
    Combined dataset from multiple cohorts for training on pooled data.
    Maintains cohort labels for potential batch effect handling.
    """
    
    def __init__(self,
                 tcga_expression: pd.DataFrame,
                 tcga_survival: pd.DataFrame,
                 orien_expression: pd.DataFrame, 
                 orien_survival: pd.DataFrame):
        """
        Create combined dataset from TCGA and ORIEN.
        Adds cohort indicator for potential cohort-aware training.
        """
        # Create individual datasets
        self.tcga_dataset = SurvivalDataset(tcga_expression, tcga_survival)
        self.orien_dataset = SurvivalDataset(orien_expression, orien_survival)
        
        # Combine features and labels
        self.X = np.vstack([self.tcga_dataset.X, self.orien_dataset.X])
        self.y_time = np.hstack([self.tcga_dataset.y_time, self.orien_dataset.y_time])
        self.y_event = np.hstack([self.tcga_dataset.y_event, self.orien_dataset.y_event])
        
        # Cohort indicators (0=TCGA, 1=ORIEN)
        self.cohort = np.hstack([
            np.zeros(len(self.tcga_dataset)),
            np.ones(len(self.orien_dataset))
        ])
        
        # Combined sample IDs
        self.sample_ids = self.tcga_dataset.sample_ids + self.orien_dataset.sample_ids
        self.n_features = self.X.shape[1]
        
        logger.info(f"Combined dataset: {len(self)} samples ({len(self.tcga_dataset)} TCGA + {len(self.orien_dataset)} ORIEN)")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'features': torch.from_numpy(self.X[idx].astype(np.float32)),
            'time': torch.tensor(self.y_time[idx], dtype=torch.float32),
            'event': torch.tensor(self.y_event[idx], dtype=torch.float32),
            'cohort': torch.tensor(self.cohort[idx], dtype=torch.long),
            'sample_id': self.sample_ids[idx]
        }


def create_survival_dataloaders(
    expression_df: pd.DataFrame,
    survival_df: pd.DataFrame,
    batch_size: int = 32,
    valid_size: float = 0.2,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for survival analysis.
    
    Returns:
        train_loader, valid_loader
    """
    # Create dataset
    dataset = SurvivalDataset(expression_df, survival_df)
    
    # Split
    train_dataset, valid_dataset = dataset.create_train_valid_split(
        valid_size=valid_size,
        random_seed=random_seed
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader


# Test function
if __name__ == "__main__":
    import yaml
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    tcga_expr = pd.read_csv("data/processed/tcga_preprocessed.csv", index_col=0)
    orien_expr = pd.read_csv("data/processed/orien_preprocessed.csv", index_col=0)
    surv_tcga = pd.read_csv("data/processed/surv_tcga_harmonized.csv", index_col=0)
    surv_orien = pd.read_csv("data/processed/surv_orien_harmonized.csv", index_col=0)
    
    # Test individual dataset
    print("\n" + "="*60)
    print("Testing TCGA Dataset")
    tcga_dataset = SurvivalDataset(tcga_expr, surv_tcga)
    sample = tcga_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Feature shape: {sample['features'].shape}")
    print(f"Time: {sample['time']:.2f}, Event: {sample['event']}")
    
    # Test combined dataset
    print("\n" + "="*60)
    print("Testing Combined Dataset")
    combined = CombinedSurvivalDataset(tcga_expr, surv_tcga, orien_expr, surv_orien)
    sample = combined[0]
    print(f"Combined dataset size: {len(combined)}")
    print(f"Cohort distribution: TCGA={(combined.cohort==0).sum()}, ORIEN={(combined.cohort==1).sum()}")