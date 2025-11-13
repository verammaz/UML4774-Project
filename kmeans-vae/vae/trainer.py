"""
Trainer class for VAE training with proper training loop.
"""

import os
import torch
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from tqdm import tqdm

from vae.utils import CfgNode as CN


class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        C.device = 'auto'
        C.num_workers = 4
        C.batch_size = 64
        C.lr = 3e-4
        C.epochs = 50
        C.optimizer = 'adam'
        C.weight_decay = 0.0
        C.beta1 = 0.9
        C.beta2 = 0.999
        C.eps = 1e-8
        C.grad_norm_clip = 1.0  # Gradient clipping
        return C

    def __init__(self, config, model, train_dataset, val_dataset=None):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.callbacks = defaultdict(list)

        # Determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else \
                         'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = config.device
        
        print(f"Training on device: {self.device}")
        self.model = self.model.to(self.device)
        
        self.iter_num = 0
        self.epoch_num = 0

    def add_callback(self, onevent: str, callback):
        """Add a callback function for a specific event."""
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        """Set a single callback function for a specific event."""
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        """Execute all callbacks for a specific event."""
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def get_optimizer(self, model, config):
        """
        Select optimizer dynamically with given hyperparameters.
        
        Args:
            model: The model to optimize
            config: Configuration containing optimizer settings
            
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        if config.optimizer.lower() == "adamw":
            return torch.optim.AdamW(
                model.parameters(), 
                lr=config.lr, 
                weight_decay=config.weight_decay, 
                betas=(config.beta1, config.beta2), 
                eps=config.eps
            )
        elif config.optimizer.lower() == "rmsprop":
            return torch.optim.RMSprop(
                model.parameters(), 
                lr=config.lr, 
                weight_decay=config.weight_decay, 
                eps=config.eps
            )
        else:  # Default to Adam
            return torch.optim.Adam(
                model.parameters(), 
                lr=config.lr, 
                weight_decay=config.weight_decay, 
                betas=(config.beta1, config.beta2), 
                eps=config.eps
            )

    def train_epoch(self, model, dataloader, device):
        """
        Train for one epoch.
        
        Args:
            model: VAE model
            dataloader: Training data loader
            device: Device to train on
            
        Returns:
            dict: Average losses for the epoch
        """
        model.train()
        total = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        
        for x_batch, _ in tqdm(dataloader, desc=f"Epoch {self.epoch_num}", leave=False):
            x_batch = x_batch.to(device)
            
            # Forward pass and compute loss
            model.optimizer.zero_grad()
            recon, mu, logvar = model(x_batch)
            loss, recon_loss, kl_loss = model.compute_loss(x_batch, recon, mu, logvar)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_norm_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_norm_clip)
            
            model.optimizer.step()
            
            # Accumulate losses
            total["loss"] += loss.item()
            total["recon"] += recon_loss.item()
            total["kl"] += kl_loss.item()
            
            self.iter_num += 1
            # self.trigger_callbacks('on_batch_end')
        
        # Average over batches
        n = len(dataloader)
        return {k: v / n for k, v in total.items()}

    @torch.no_grad()
    def validate_epoch(self, model, dataloader, device):
        """
        Validate for one epoch.
        
        Args:
            model: VAE model
            dataloader: Validation data loader
            device: Device to validate on
            
        Returns:
            dict: Average losses for validation
        """
        model.eval()
        total = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            
            # Forward pass
            recon, mu, logvar = model(x_batch)
            loss, recon_loss, kl_loss = model.compute_loss(x_batch, recon, mu, logvar)
            
            # Accumulate losses
            total["loss"] += loss.item()
            total["recon"] += recon_loss.item()
            total["kl"] += kl_loss.item()
        
        # Average over batches
        n = len(dataloader)
        return {k: v / n for k, v in total.items()}

    def run(self):
        """
        Main training loop.
        """
        model, config = self.model, self.config
        
        # Initialize optimizer
        model.optimizer = self.get_optimizer(model, config)
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=(self.device == 'cuda')
        )
        
        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=(self.device == 'cuda')
            )
        
        # Trigger start callback
        self.trigger_callbacks('on_train_start')
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(config.epochs):
            self.epoch_num = epoch
            
            # Train for one epoch
            train_stats = self.train_epoch(model, train_loader, self.device)
            
            # Validate
            val_stats = None
            if val_loader is not None:
                val_stats = self.validate_epoch(model, val_loader, self.device)
            
            # Print progress
            log_str = (
                f"Epoch {epoch+1}/{config.epochs} | "
                f"Train Loss: {train_stats['loss']:.4f} "
                f"(Recon: {train_stats['recon']:.4f}, KL: {train_stats['kl']:.4f})"
            )
            
            if val_stats is not None:
                log_str += (
                    f" | Val Loss: {val_stats['loss']:.4f} "
                    f"(Recon: {val_stats['recon']:.4f}, KL: {val_stats['kl']:.4f})"
                )
                
                # Track best validation loss
                if val_stats['loss'] < best_val_loss:
                    best_val_loss = val_stats['loss']
                    log_str += " [BEST]"
            
            print(log_str)
            
            # Store stats for callbacks
            self.train_stats = train_stats
            self.val_stats = val_stats
            
            # Trigger epoch end callback
            self.trigger_callbacks('on_epoch_end')
        
        # Trigger end callback
        self.trigger_callbacks('on_train_end')
        
        print(f"\nTraining completed!")
        if val_loader is not None:
            print(f"Best validation loss: {best_val_loss:.4f}")