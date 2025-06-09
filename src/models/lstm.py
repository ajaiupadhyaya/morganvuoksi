"""
LSTM model for time series forecasting.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ..utils.config import get_config
from ..utils.logging import setup_logger
from .base import BaseModel

logger = setup_logger(__name__)

class LSTMModel(nn.Module):
    """LSTM neural network for time series forecasting."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            hidden: Initial hidden state
            
        Returns:
            Output tensor and hidden state
        """
        batch_size = x.size(0)
        
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        else:
            h0, c0 = hidden
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        
        return out, (hn, cn)

class LSTM(BaseModel):
    """LSTM model wrapper with training and prediction functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LSTM model.
        
        Args:
            config: Optional model configuration
        """
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def _prepare_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        sequence_length: int = 10
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare data for LSTM.
        
        Args:
            X: Features
            y: Optional target variable
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X_tensor, y_tensor)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - sequence_length):
            X_sequences.append(X[i:i + sequence_length])
            if y is not None:
                y_sequences.append(y[i + sequence_length])
        
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        y_tensor = torch.FloatTensor(y_sequences).to(self.device) if y is not None else None
        
        return X_tensor, y_tensor
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> 'LSTM':
        """
        Fit LSTM model.
        
        Args:
            X: Features
            y: Target variable
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        # Get model parameters
        input_size = X.shape[1] if isinstance(X, pd.DataFrame) else X.shape[1]
        hidden_size = self.config.get('hidden_size', 128)
        num_layers = self.config.get('num_layers', 2)
        output_size = 1
        dropout = self.config.get('dropout', 0.2)
        sequence_length = self.config.get('sequence_length', 10)
        batch_size = self.config.get('batch_size', 32)
        epochs = self.config.get('epochs', 100)
        learning_rate = self.config.get('learning_rate', 0.001)
        
        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y, sequence_length)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output, _ = self.model(batch_X)
                loss = criterion(output.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        sequence_length = self.config.get('sequence_length', 10)
        X_tensor, _ = self._prepare_data(X, sequence_length=sequence_length)
        
        self.model.eval()
        with torch.no_grad():
            predictions, _ = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make probability predictions.
        
        Args:
            X: Features
            
        Returns:
            Probability predictions
        """
        raise NotImplementedError("LSTM does not support probability predictions")
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Saved model to {path}")
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path)
        self.config = checkpoint['config']
        
        # Initialize model
        input_size = self.config.get('input_size')
        hidden_size = self.config.get('hidden_size', 128)
        num_layers = self.config.get('num_layers', 2)
        output_size = 1
        dropout = self.config.get('dropout', 0.2)
        
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted = True
        logger.info(f"Loaded model from {path}") 