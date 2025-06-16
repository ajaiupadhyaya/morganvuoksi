"""
Model-Agnostic Meta-Learning (MAML) implementation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class MAML:
    """Model-Agnostic Meta-Learning for quick adaptation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = self._create_model()
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=config.get('meta_lr', 1e-3))
        self.inner_lr = config.get('inner_lr', 1e-2)
        self.num_inner_steps = config.get('num_inner_steps', 5)
    
    def _create_model(self) -> nn.Module:
        """Create base model."""
        input_dim = self.config.get('input_dim', 10)
        hidden_dim = self.config.get('hidden_dim', 64)
        output_dim = self.config.get('output_dim', 1)
        
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def _adapt(self, X: torch.Tensor, y: torch.Tensor) -> nn.Module:
        """
        Adapt model to new task.
        
        Args:
            X: Task features
            y: Task targets
            
        Returns:
            Adapted model
        """
        # Create copy of model
        adapted_model = type(self.model)(*self.model.children())
        adapted_model.load_state_dict(self.model.state_dict())
        
        # Inner loop optimization
        optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        criterion = nn.MSELoss()
        
        for _ in range(self.num_inner_steps):
            optimizer.zero_grad()
            y_pred = adapted_model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def meta_train(self, tasks: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """
        Meta-train the model.
        
        Args:
            tasks: List of (X, y) tuples for each task
            
        Returns:
            Meta-loss
        """
        meta_loss = 0
        
        for X, y in tasks:
            # Adapt model to task
            adapted_model = self._adapt(X, y)
            
            # Compute meta-loss
            y_pred = adapted_model(X)
            loss = nn.MSELoss()(y_pred, y)
            meta_loss += loss
        
        # Average meta-loss
        meta_loss /= len(tasks)
        
        # Meta-optimization
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def predict(self, X: torch.Tensor, adapt: bool = True) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            X: Features
            adapt: Whether to adapt model before prediction
            
        Returns:
            Predictions
        """
        if adapt:
            # Create dummy target for adaptation
            y = torch.zeros(len(X), 1)
            model = self._adapt(X, y)
        else:
            model = self.model
        
        return model(X)
    
    def save(self, path: str) -> None:
        """Save model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict()
        }, path)
        logger.info(f"Saved MAML model to {path}")
    
    def load(self, path: str) -> None:
        """Load model."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        logger.info(f"Loaded MAML model from {path}") 
