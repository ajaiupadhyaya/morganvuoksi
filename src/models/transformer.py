"""
Transformer model for sequence learning and macro signal interaction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import math
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(0)]

class Transformer(nn.Module):
    """Transformer model for financial time series."""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.d_model = config.get('d_model', 64)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.dim_feedforward = config.get('dim_feedforward', 256)
        self.dropout = config.get('dropout', 0.1)
        self.input_dim = config.get('input_dim', 10)
        self.output_dim = config.get('output_dim', 1)
        self.max_seq_len = config.get('max_seq_len', 100)
        
        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_feedforward, self.output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer."""
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x, mask=mask)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

class TransformerModel:
    """Wrapper class for transformer model with training logic."""
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Transformer(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        self.criterion = nn.MSELoss()
    
    def prepare_data(self, data: np.ndarray, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare sequences for training."""
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len, 0])  # Predict next price
        return torch.FloatTensor(X).to(self.device), torch.FloatTensor(y).to(self.device)
    
    def train(self, data: np.ndarray, n_epochs: int, batch_size: int = 32) -> List[float]:
        """Train the model."""
        seq_len = self.config.get('seq_len', 20)
        X, y = self.prepare_data(data, seq_len)
        n_samples = len(X)
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            n_batches = 0
            
            # Shuffle data
            indices = torch.randperm(n_samples)
            X = X[indices]
            y = y[indices]
            
            for i in range(0, n_samples, batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.criterion(output.squeeze(), batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Update learning rate
            avg_loss = epoch_loss / n_batches
            self.scheduler.step(avg_loss)
            losses.append(avg_loss)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        self.model.eval()
        seq_len = self.config.get('seq_len', 20)
        X, _ = self.prepare_data(data, seq_len)
        
        with torch.no_grad():
            predictions = self.model(X)
        
        return predictions.cpu().numpy()
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict']) 
