import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal, List
from vae.utils import CfgNode as CN


class VAE(nn.Module):
    """
    Variational AutoEncoder
    - Encoder: X -> (mu, log_var)
    - Latent sampling: z ~ N(mu, log_var)
    - Decoder: z -> X_reconstructed
    """

    def __init__(
            self,
            input_dim: int,
            latent_dim: int = 10,
            hidden_dims: List[int] = [128, 64],
            likelihood: Literal["gaussian", "bernoulli"] = "gaussian",
            beta: float = 1.0,  # KL weighting (beta-VAE)
            seed: Optional[int] = None,
            activation: Optional[str] = 'LeakyReLU'):
        """
        Args:
            input_dim: Number of input features
            latent_dim: Dimension of latent space
            hidden_dims: Hidden layer sizes (encoder and decoder mirror each other)
            likelihood: "gaussian" for continuous data, "bernoulli" for binary
            beta: Weight on KL term (beta=1 is standard VAE, beta>1 encourages disentanglement)
            seed: Random seed for reproducibility
            activation: Activation function ('LeakyReLU' or 'ReLU')
        """

        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.likelihood = likelihood
        self.beta = beta
        self.activation = activation

        self.optimizer = None  # Will be set within Trainer
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Initialize encoder
        self._init_encoder()

        # Latent Space
        self.mean = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.logvar = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        # Initialize decoder
        self._init_decoder()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _init_encoder(self):
        """Build encoder network."""
        encoder_layers = []
        
        layer_sizes = [self.input_dim] + list(self.hidden_dims)

        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # No activation after last layer
                if self.activation == 'LeakyReLU':
                    encoder_layers.append(nn.LeakyReLU(0.2))
                else:
                    encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

    def _init_decoder(self):
        """Build decoder network."""
        decoder_layers = []
        layer_sizes = [self.latent_dim] + list(reversed(self.hidden_dims))

        for i in range(len(layer_sizes) - 1):
            decoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # No activation after last layer
                if self.activation == 'LeakyReLU':
                    decoder_layers.append(nn.LeakyReLU(0.2))
                else:
                    decoder_layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_layers)
        self.output_layer = nn.Linear(layer_sizes[-1], self.input_dim)

    def encode(self, x):
        """
        Encode input to latent parameters.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            tuple: (mu, logvar) each of shape (batch_size, latent_dim)
        """
        h = self.encoder(x)
        return self.mean(h), self.logvar(h)

    def decode(self, z):
        """
        Decode latent vector to reconstruction.
        
        Args:
            z: Latent tensor (batch_size, latent_dim)
            
        Returns:
            tensor: Reconstruction (batch_size, input_dim)
        """
        h = self.decoder(z)
        return self.output_layer(h)

    def reparametrize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Full forward pass through VAE.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            tuple: (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def compute_loss(self, x, recon, mu, logvar):
        """
        Compute VAE loss = reconstruction_loss + beta * KL_divergence
        
        Args:
            x: Original input
            recon: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            tuple: (total_loss, recon_loss, kl_loss)
        """
        batch_size = x.size(0)
        
        # Reconstruction loss
        if self.likelihood == "gaussian":
            # MSE loss
            recon_loss = F.mse_loss(recon, x, reduction="sum") / batch_size
        elif self.likelihood == "bernoulli":
            # Binary cross-entropy 
            recon_loss = F.binary_cross_entropy_with_logits(recon, x, reduction="sum") / batch_size
        else:
            raise ValueError(f"Unknown likelihood: {self.likelihood}")

        # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def train_step(self, x):
        """
        Single training step (kept for backward compatibility).
        
        Args:
            x: Input batch
            
        Returns:
            dict: Loss statistics
        """
        self.optimizer.zero_grad()
        recon, mu, logvar = self.forward(x)
        loss, recon_loss, kl_loss = self.compute_loss(x, recon, mu, logvar)
        loss.backward()
        self.optimizer.step()
        return {
            "loss": loss.item(),
            "recon": recon_loss.item(),
            "kl": kl_loss.item()
        }

    @torch.no_grad()
    def reconstruct(self, x):
        """
        Reconstruct input (using mean of latent distribution, no sampling).
        
        Args:
            x: Input tensor
            
        Returns:
            tensor: Reconstructed input
        """
        mu, _ = self.encode(x)
        recon = self.decode(mu)
        if self.likelihood == "bernoulli":
            recon = torch.sigmoid(recon)
        return recon

    @torch.no_grad()
    def sample(self, n_samples: int, device=None):
        """
        Generate new samples from the prior p(z) = N(0, I).
        
        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            tensor: Generated samples
        """
        if device is None:
            device = next(self.parameters()).device
        
        z = torch.randn(n_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        if self.likelihood == "bernoulli":
            samples = torch.sigmoid(samples)
        return samples

    @torch.no_grad()
    def encode_data(self, x):
        """
        Encode data to latent space (returns mean, no sampling).
        
        Args:
            x: Input tensor
            
        Returns:
            tensor: Latent representations (means)
        """
        mu, _ = self.encode(x)
        return mu

    @staticmethod
    def get_default_config():
        """Get default configuration for VAE."""
        C = CN()
        C.name = 'vae'
        C.latent_dim = 10
        C.hidden_dims = [128, 64]
        C.likelihood = 'gaussian'
        C.kl_beta = 1.0
        C.seed = 42
        C.activation = 'LeakyReLU'
        return C