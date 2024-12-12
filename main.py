"""
Efficient Transformer for Tabular Time Series
-------------------------------------------
A memory-efficient transformer implementation specifically designed for tabular time series data.
Includes production features like logging, error handling, and performance optimizations.

Author: Claude
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Union
from dataclasses import dataclass
from loguru import logger
import math
import warnings
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl

# Configure loguru logger
logger.add(
    "transformer.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

@dataclass
class TransformerConfig:
    """Configuration for the transformer model."""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    max_seq_length: int = 1000
    feature_dim: int = 64
    num_features: int = 10
    compression_factor: int = 4
    attention_window: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class LinearSparseAttention(nn.Module):
    """Memory-efficient sparse attention implementation."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.window_size = config.attention_window
        
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.o_proj = nn.Linear(config.d_model, config.d_model)
        
        self.cache = None
        logger.info(f"Initialized LinearSparseAttention with window size {self.window_size}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with sliding window attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Compute attention scores with sliding window
        attention = torch.zeros(batch_size, self.n_heads, seq_len, seq_len, 
                              device=x.device, dtype=x.dtype)
        
        for i in range(0, seq_len, self.window_size):
            end_idx = min(i + self.window_size, seq_len)
            
            # Compute attention for the current window
            scores = torch.matmul(
                q[:, i:end_idx].transpose(1, 2),
                k[:, max(0, i-self.window_size):end_idx].transpose(1, 2).transpose(2, 3)
            )
            scores = scores / math.sqrt(self.head_dim)
            
            if mask is not None:
                window_mask = mask[:, i:end_idx, max(0, i-self.window_size):end_idx]
                scores = scores.masked_fill(~window_mask, float('-inf'))
            
            attention[:, :, i:end_idx, max(0, i-self.window_size):end_idx] = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, v.transpose(1, 2)).transpose(1, 2)
        out = self.o_proj(out.reshape(batch_size, seq_len, self.d_model))
        
        return out

class GatedMemoryUnit(nn.Module):
    """Gated memory unit for controlling information flow."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model * 2, config.d_model)
        self.update = nn.Linear(config.d_model * 2, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
        logger.info("Initialized GatedMemoryUnit")
    
    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the gated memory unit.
        
        Args:
            x: Input tensor
            memory: Current memory state
            
        Returns:
            Tuple of (output tensor, updated memory)
        """
        combined = torch.cat([x, memory], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        update = self.update(combined)
        
        new_memory = gate * update + (1 - gate) * memory
        output = self.layer_norm(x + new_memory)
        
        return output, new_memory

class TemporalCompression(nn.Module):
    """Temporal compression module for reducing sequence length."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=config.compression_factor,
            stride=config.compression_factor
        )
        self.norm = nn.LayerNorm(config.d_model)
        logger.info(f"Initialized TemporalCompression with factor {config.compression_factor}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress the temporal dimension of the input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Compressed tensor
        """
        # Reshape for 1D convolution
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.norm(x)

class EfficientTransformer(nn.Module):
    """Main transformer model with memory-efficient architecture."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.feature_embedding = nn.Linear(config.num_features, config.feature_dim)
        self.temporal_embedding = nn.Embedding(config.max_seq_length, config.feature_dim)
        self.embedding_proj = nn.Linear(config.feature_dim * 2, config.d_model)
        
        # Main components
        self.attention_layers = nn.ModuleList([
            LinearSparseAttention(config) for _ in range(config.n_layers)
        ])
        self.memory_units = nn.ModuleList([
            GatedMemoryUnit(config) for _ in range(config.n_layers)
        ])
        self.temporal_compression = TemporalCompression(config)
        
        # Output layers
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.num_features)
        
        self.dropout = nn.Dropout(config.dropout)
        self._init_weights()
        
        logger.info(f"Initialized EfficientTransformer with config: {config}")
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            timestamps: Tensor of timestamp indices
            mask: Optional attention mask
            
        Returns:
            Output predictions
        """
        try:
            # Input validation
            if x.dim() != 3:
                raise ValueError(f"Expected 3D input tensor, got shape {x.shape}")
            if x.size(-1) != self.config.num_features:
                raise ValueError(
                    f"Expected {self.config.num_features} features, got {x.size(-1)}"
                )
            
            # Compute embeddings
            feature_emb = self.feature_embedding(x)
            temp_emb = self.temporal_embedding(timestamps)
            embeddings = self.embedding_proj(
                torch.cat([feature_emb, temp_emb], dim=-1)
            )
            embeddings = self.dropout(embeddings)
            
            # Initialize memory
            batch_size, seq_len, _ = embeddings.shape
            memory = torch.zeros(
                batch_size, seq_len, self.config.d_model,
                device=embeddings.device
            )
            
            # Process through transformer layers
            hidden_states = embeddings
            for attention, memory_unit in zip(self.attention_layers, self.memory_units):
                # Use gradient checkpointing for memory efficiency
                attention_output = checkpoint(
                    attention,
                    hidden_states,
                    mask
                )
                hidden_states, memory = memory_unit(attention_output, memory)
            
            # Compress and generate output
            compressed = self.temporal_compression(hidden_states)
            output = self.output_norm(compressed)
            predictions = self.output_proj(output)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate predictions with no gradient computation.
        
        Args:
            x: Input tensor
            timestamps: Timestamp indices
            mask: Optional attention mask
            
        Returns:
            Model predictions
        """
        self.eval()
        try:
            return self(x, timestamps, mask)
        finally:
            self.train()
    
    def get_attention_patterns(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention patterns for analysis.
        
        Args:
            x: Input tensor
            timestamps: Timestamp indices
            
        Returns:
            Dictionary of attention patterns
        """
        patterns = {}
        with torch.no_grad():
            # Forward pass collecting attention weights
            for i, attention in enumerate(self.attention_layers):
                patterns[f"layer_{i}"] = attention(
                    x, timestamps
                ).detach().cpu()
        return patterns

# Training utilities
def create_transformer(
    num_features: int,
    max_seq_length: int,
    d_model: int = 256,
    **kwargs
) -> EfficientTransformer:
    """
    Factory function to create a transformer instance.
    
    Args:
        num_features: Number of input features
        max_seq_length: Maximum sequence length
        d_model: Model dimension
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured transformer model
    """
    config = TransformerConfig(
        num_features=num_features,
        max_seq_length=max_seq_length,
        d_model=d_model,
        **kwargs
    )
    model = EfficientTransformer(config)
    logger.info(f"Created transformer model with {sum(p.numel() for p in model.parameters())} parameters")
    return model

class EfficientTransformerLightning(pl.LightningModule):
    """PyTorch Lightning module for training the transformer."""
    
    def __init__(self, model: EfficientTransformer, learning_rate: float = 1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        
    def forward(self, x, timestamps, mask=None):
        return self.model(x, timestamps, mask)
    
    def training_step(self, batch, batch_idx):
        x, timestamps, y = batch
        y_hat = self(x, timestamps)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# Example usage
if __name__ == "__main__":
    # Set up logging
    logger.add("transformer_training.log")
    
    try:
        # Create model
        model = create_transformer(
            num_features=10,
            max_seq_length=1000,
            d_model=256
        )
        
        # Example input
        batch_size = 32
        seq_length = 100
        x = torch.randn(batch_size, seq_length, 10)
        timestamps = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
        
        # Generate predictions
        with torch.no_grad():
            predictions = model.predict(x, timestamps)
        
        logger.info(f"Generated predictions of shape {predictions.shape}")
        
    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}")
        raise
