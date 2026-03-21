"""
Custom batch samplers for survival analysis.

Ensures each batch contains events for valid Cox loss computation.

Based on:
- Katzman et al. (2018), "DeepSurv: personalized treatment recommender system"
- Kvamme et al. (2019), "Time-to-Event Prediction with Neural Networks and Cox Regression"
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Iterator, List
import logging

logger = logging.getLogger(__name__)


class StratifiedBatchSampler(Sampler):
    """
    Batch sampler that ensures each batch contains a minimum number of events.
    
    Critical for Cox loss training where batches without events cannot
    contribute to gradient updates, leading to wasted computation and
    training instability.
    
    Strategy:
    1. Separate samples into event and censored groups
    2. For each batch, first add required number of events
    3. Fill remainder with censored samples
    4. Shuffle within batch for randomness
    
    Args:
        events: Binary array/tensor of event indicators (1=event, 0=censored)
        batch_size: Target batch size
        min_events_per_batch: Minimum events per batch (default: 1)
        shuffle: Whether to shuffle order (default: True)
        drop_last: Whether to drop last incomplete batch (default: False)
    """
    
    def __init__(
        self,
        events: np.ndarray,
        batch_size: int,
        min_events_per_batch: int = 2,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        # Convert to numpy if tensor
        if torch.is_tensor(events):
            events = events.cpu().numpy()
        
        self.events = np.array(events)
        self.batch_size = batch_size
        self.min_events_per_batch = min_events_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Separate event and censored indices
        self.event_indices = np.where(self.events == 1)[0]
        self.censored_indices = np.where(self.events == 0)[0]
        
        self.n_events = len(self.event_indices)
        self.n_censored = len(self.censored_indices)
        self.n_total = len(self.events)
        
        # Validate
        if self.n_events < self.min_events_per_batch:
            raise ValueError(
                f"Dataset has only {self.n_events} events, but "
                f"min_events_per_batch={self.min_events_per_batch}. "
                f"Reduce min_events_per_batch or check your data."
            )
        
        # Calculate number of batches
        if self.drop_last:
            self.n_batches = self.n_total // self.batch_size
        else:
            self.n_batches = (self.n_total + self.batch_size - 1) // self.batch_size
        
        logger.info(f"StratifiedBatchSampler initialized:")
        logger.info(f"  Total samples: {self.n_total}")
        logger.info(f"  Events: {self.n_events} ({100*self.n_events/self.n_total:.1f}%)")
        logger.info(f"  Censored: {self.n_censored} ({100*self.n_censored/self.n_total:.1f}%)")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Min events per batch: {self.min_events_per_batch}")
        logger.info(f"  Number of batches: {self.n_batches}")
    
    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices if requested
        if self.shuffle:
            event_indices = np.random.permutation(self.event_indices)
            censored_indices = np.random.permutation(self.censored_indices)
        else:
            event_indices = self.event_indices.copy()
            censored_indices = self.censored_indices.copy()
        
        # Calculate events per batch to ensure minimum and balanced distribution
        events_per_batch = max(
            self.min_events_per_batch,
            int(np.ceil(self.n_events / self.n_batches))
        )
        
        event_ptr = 0
        censored_ptr = 0
        batch_count = 0
        
        while batch_count < self.n_batches:
            batch = []
            
            # Step 1: Add events (guaranteed minimum)
            n_events_available = self.n_events - event_ptr
            n_events_this_batch = min(events_per_batch, n_events_available)
            
            if n_events_this_batch > 0:
                batch.extend(event_indices[event_ptr:event_ptr + n_events_this_batch])
                event_ptr += n_events_this_batch
            
            # Step 2: Fill remainder with censored samples
            n_censored_needed = self.batch_size - len(batch)
            n_censored_available = self.n_censored - censored_ptr
            n_censored_this_batch = min(n_censored_needed, n_censored_available)
            
            if n_censored_this_batch > 0:
                batch.extend(censored_indices[censored_ptr:censored_ptr + n_censored_this_batch])
                censored_ptr += n_censored_this_batch
            
            # Step 3: Handle last batch
            if len(batch) == 0:
                break
            
            if self.drop_last and len(batch) < self.batch_size:
                break
            
            # Step 4: Shuffle within batch for randomness
            if self.shuffle:
                np.random.shuffle(batch)
            
            yield batch
            batch_count += 1
    
    def __len__(self) -> int:
        return self.n_batches


class AdaptiveStratifiedBatchSampler(StratifiedBatchSampler):
    """
    Advanced version that adapts min_events_per_batch based on dataset size.
    
    Automatically calculates optimal minimum events per batch to ensure:
    1. Every batch has at least 1 event
    2. Events are distributed roughly evenly across batches
    3. Batch sizes remain reasonable
    
    This is useful when you don't want to manually tune min_events_per_batch.
    """
    
    def __init__(
        self,
        events: np.ndarray,
        batch_size: int,
        target_event_rate: float = None,  # If None, use dataset event rate
        shuffle: bool = True,
        drop_last: bool = False
    ):
        # Calculate optimal min_events_per_batch
        n_events = np.sum(events)
        n_total = len(events)
        dataset_event_rate = n_events / n_total
        
        # Use target or dataset event rate
        if target_event_rate is None:
            target_event_rate = dataset_event_rate
        
        # Calculate expected events per batch
        expected_events_per_batch = batch_size * target_event_rate
        
        # Set minimum to at least 1, but prefer expected
        min_events = max(1, int(np.floor(expected_events_per_batch)))
        
        logger.info(f"AdaptiveStratifiedBatchSampler:")
        logger.info(f"  Dataset event rate: {dataset_event_rate:.1%}")
        logger.info(f"  Target event rate: {target_event_rate:.1%}")
        logger.info(f"  Expected events per batch: {expected_events_per_batch:.1f}")
        logger.info(f"  Setting min_events_per_batch: {min_events}")
        
        super().__init__(
            events=events,
            batch_size=batch_size,
            min_events_per_batch=min_events,
            shuffle=shuffle,
            drop_last=drop_last
        )


# Testing code
if __name__ == "__main__":
    print("="*60)
    print("Testing StratifiedBatchSampler")
    print("="*60)
    
    # Create sample data: 100 samples, 30% event rate
    np.random.seed(42)
    events = np.random.binomial(1, 0.3, 100)
    
    print(f"\nDataset: {len(events)} samples, {events.sum()} events ({100*events.mean():.1f}%)")
    
    # Test basic sampler
    sampler = StratifiedBatchSampler(
        events=events,
        batch_size=32,
        min_events_per_batch=2,
        shuffle=False
    )
    
    print(f"\nBatch composition:")
    for i, batch in enumerate(sampler):
        batch_events = events[batch].sum()
        print(f"  Batch {i+1}: {len(batch)} samples, {batch_events} events "
              f"({100*batch_events/len(batch):.1f}%)")
    
    # Test adaptive sampler
    print(f"\n{'='*60}")
    print("Testing AdaptiveStratifiedBatchSampler")
    print("="*60)
    
    adaptive_sampler = AdaptiveStratifiedBatchSampler(
        events=events,
        batch_size=32,
        shuffle=False
    )
    
    print(f"\nBatch composition:")
    for i, batch in enumerate(adaptive_sampler):
        batch_events = events[batch].sum()
        print(f"  Batch {i+1}: {len(batch)} samples, {batch_events} events "
              f"({100*batch_events/len(batch):.1f}%)")
    
    print(f"\n{'='*60}")
    print("All tests passed!")
    print("="*60)