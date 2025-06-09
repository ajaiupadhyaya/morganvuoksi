"""
Base model class for all ML models.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from ..utils.config import get_config
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class BaseModel(ABC):
    """Base class for all ML models with common functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model with configuration.
        
        Args:
            config: Optional model-specific configuration
        """
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'BaseModel':
        """
        Fit the model to the data.
        
        Args:
            X: Features
            y: Target variable
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.Series, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Make probability predictions.
        
        Args:
            X: Features
            
        Returns:
            Probability predictions
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        raise NotImplementedError
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.config
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        self.config.update(params)
        return self 