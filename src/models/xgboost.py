"""
XGBoost model for time series forecasting.
"""
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
import xgboost as xgb
from ..utils.config import get_config
from ..utils.logging import setup_logger
from .base import BaseModel

logger = setup_logger(__name__)

class XGBoost(BaseModel):
    """XGBoost model wrapper with training and prediction functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model.
        
        Args:
            config: Optional model configuration
        """
        super().__init__(config)
        self.model = None
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> 'XGBoost':
        """
        Fit XGBoost model.
        
        Args:
            X: Features
            y: Target variable
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        # Get model parameters
        params = {
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'n_estimators': self.config.get('n_estimators', 100),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse'
        }
        
        # Initialize model
        self.model = xgb.XGBRegressor(**params)
        
        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Log feature importance
        if isinstance(X, pd.DataFrame):
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            logger.info(f"Feature importance:\n{importance}")
        
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
        
        return self.model.predict(X)
    
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
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict_proba(X)
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        self.model.save_model(path)
        logger.info(f"Saved model to {path}")
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        self.is_fitted = True
        logger.info(f"Loaded model from {path}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = pd.DataFrame({
            'feature': self.model.get_booster().feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance 