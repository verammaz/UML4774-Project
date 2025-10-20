"""
Trains VAE with optional W&B logging.
"""

import os
import sys
import json
import glob

import torch
from torch.utils.data import TensorDataset

from vae.model import VAE
from vae.trainer import Trainer
from data.data_io import load_and_split
from vae.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.out_dir = './out'

    # wandb
    C.wandb = CN()
    C.wandb.enabled = False
    C.wandb.project = 'kmeans-vae'
    C.wandb.entity = None  # Your wandb username/team
    C.wandb.name = None  # Run name (auto-generated if None)
    C.wandb.tags = []  # List of tags
    C.wandb.notes = ''  # Run notes/description
    C.wandb.log_freq = 10  # Log every N batches

    # data
    C.data = CN()
    C.data.data_dir = None

    # model
    C.model = VAE.get_default_config()

    # trainer
    C.trainer = Trainer.get_default_config()

    return C

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Get and merge config
    config = get_config()
    config.merge_from_args(sys.argv[1:])

    # Check data directory is provided
    if config.data.data_dir is None:
        print()
        sys.exit() 
    
    print("Configuration:")
    print(config)
    print()
    
    # Set seed
    set_seed(config.system.seed)
    
    # Load training data
    print(f"Loading dataset from {config.data.data_dir}")
    data = load_and_split(
        config.data.data_dir,
        splits=(0.85, 0.0, 0.15),
        seed=config.system.seed,
        normalize=True,
        save_manifest=True
    )
    
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.long)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)
    
    print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set:  {X_test.shape[0]} samples")
    print(f"Classes:   {len(torch.unique(y_train))}")
    print()


    # Auto-generate model name if not provided
    if config.model.name is None:
        dataset_name = 'gaus' if 'gaussian' in config.data.data_dir else 'ber'
        k = len(torch.unique(y_train))
        config.model.name = f"vae_{dataset_name}_i{X_train.shape[1]}_k{k}_z{config.model.latent_dim}_beta{config.model.kl_beta}"
    
    # Setup logging (creates out_dir)
    setup_logging(config)

    # Initialize wandb if enabled
    wandb_run = None
    if config.wandb.enabled:
        try:
            import wandb
            
            # Auto-generate run name if not provided
            if config.wandb.name is None:
                config.wandb.name = config.model.name
            
            wandb_run = wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=config.wandb.name,
                tags=config.wandb.tags if config.wandb.tags else None,
                notes=config.wandb.notes if config.wandb.notes else None,
                config=config.to_dict(),
                reinit=True
            )
            print(f"W&B logging enabled: {wandb_run.url}")
            print()
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")
            print("Continuing without W&B logging.\n")
            config.wandb.enabled = False
    

    # Get model config and verify likelihood matches data type
    model_config = config.model 
    
    
    # Auto-detect likelihood from metadata
    with open(os.path.join(config.data.data_dir, "metadata.json")) as f:
        meta = json.load(f)
    data_likelihood = meta['type']  # 'gaussian' or 'bernoulli'
    if model_config.likelihood != data_likelihood:
        print(f"Info: Setting model likelihood to '{data_likelihood}' (from dataset metadata)")
        model_config.likelihood = data_likelihood
    
    print(f"Creating VAE model:")
    print(f"  Input dim:    {X_train.shape[1]}")
    print(f"  Latent dim:   {model_config.latent_dim}")
    print(f"  Hidden dims:  {model_config.hidden_dims}")
    print(f"  Likelihood:   {model_config.likelihood}")
    print(f"  Beta (KL):    {model_config.kl_beta}")
    print(f"  Activation:   {model_config.activation}")
    print()

    # Create model
    model = VAE(
        input_dim=X_train.shape[1],
        latent_dim=model_config.latent_dim,
        hidden_dims=model_config.hidden_dims,
        likelihood=model_config.likelihood,
        beta=model_config.kl_beta,
        seed=model_config.seed,
        activation=model_config.activation
    )

    # Create datasets (VAE doesn't use labels during training, but we include them for potential eval)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create trainer
    trainer = Trainer(config.trainer, model, train_dataset, val_dataset=None)
    
    # Add wandb logging callback
    if config.wandb.enabled:
        def log_to_wandb(trainer):
            # Log batch-level metrics periodically
            if trainer.iter_num % config.wandb.log_freq == 0:
                log_dict = {
                    "iter": trainer.iter_num,
                    "epoch": trainer.epoch_num,
                }
                wandb.log(log_dict, step=trainer.iter_num)
        
        def log_epoch_to_wandb(trainer):
            # Log epoch-level metrics
            log_dict = {
                "epoch": trainer.epoch_num,
                "train/loss": trainer.train_stats['loss'],
                "train/recon": trainer.train_stats['recon'],
                "train/kl": trainer.train_stats['kl'],
            }
            
            if trainer.val_stats is not None:
                log_dict.update({
                    "val/loss": trainer.val_stats['loss'],
                    "val/recon": trainer.val_stats['recon'],
                    "val/kl": trainer.val_stats['kl'],
                })
            
            wandb.log(log_dict, step=trainer.iter_num)
        
        trainer.add_callback('on_batch_end', log_to_wandb)
        trainer.add_callback('on_epoch_end', log_epoch_to_wandb)
        
        # Watch model
        wandb.watch(model, log='all', log_freq=100)
    
    # Add checkpoint saving callback
    def save_checkpoint(trainer):
        if trainer.iter_num % 500 == 0 and trainer.iter_num > 0:  # Save every 500 iterations
            checkpoint_path = os.path.join(config.system.out_dir, f"checkpoint_iter_{trainer.iter_num}.pt")
            torch.save({
                'iter': trainer.iter_num,
                'epoch': trainer.epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'config': config.to_dict(),
            }, checkpoint_path)
            print(f"  → Saved checkpoint to {checkpoint_path}")
            
            if config.wandb.enabled:
                # Save checkpoint to wandb
                wandb.save(checkpoint_path)
    
    trainer.add_callback('on_batch_end', save_checkpoint)
    
    # Train the model
    print("Starting training...")
    print("="*60)
    trainer.run()
    
    # Final evaluation on test set
    print()
    print("="*60)
    print("Test Set Evaluation")
    print("="*60)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config.trainer.batch_size,
        shuffle=False
    )
    
    test_stats = trainer.validate_epoch(model, test_loader, trainer.device)
    print(f"Test Loss:        {test_stats['loss']:.4f}")
    print(f"  Reconstruction: {test_stats['recon']:.4f}")
    print(f"  KL Divergence:  {test_stats['kl']:.4f}")
    
    # Log test metrics to wandb
    if config.wandb.enabled:
        wandb.log({
            "test/loss": test_stats['loss'],
            "test/recon": test_stats['recon'],
            "test/kl": test_stats['kl'],
        })
        
    
    # Save final model
    final_model_path = os.path.join(config.system.out_dir, config.model.name, "model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'test_stats': test_stats,
    }, final_model_path)

    print(f"\nSaved final model to {final_model_path}")
    
    if config.wandb.enabled:
        # Save final model to wandb
        wandb.save(final_model_path)

        # Save data to wandb
        data_files = glob.glob(os.path.join(config.data.data_dir, "*"))
        for fpath in data_files:
            wandb.save(fpath)
        
        # Log final summary
        wandb.run.summary["final_test_loss"] = test_stats['loss']
        wandb.run.summary["final_test_recon"] = test_stats['recon']
        wandb.run.summary["final_test_kl"] = test_stats['kl']
        
        # Finish wandb run
        wandb.finish()
        print("W&B run finished successfully!")