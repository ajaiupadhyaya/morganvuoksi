"""
ARIMA-GARCH model for time series forecasting.
"""
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from arch import arch_model
from pmdarima import auto_arima
from ..utils.config import get_config
from ..utils.logging import setup_logger
from .base import BaseModel

logger = setup_logger(__name__)

class ARIMAGARCH(BaseModel):
    """ARIMA-GARCH model wrapper with training and prediction functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ARIMA-GARCH model.
        
        Args:
            config: Optional model configuration
        """
        super().__init__(config)
        self.arima_model = None
        self.garch_model = None
        self.residuals = None
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> 'ARIMAGARCH':
        """
        Fit ARIMA-GARCH model.
        
        Args:
            X: Features (not used for ARIMA-GARCH)
            y: Target variable
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        if isinstance(y, pd.Series):
            y = y.values
        
        # Get model parameters
        max_p = self.config.get('max_p', 5)
        max_d = self.config.get('max_d', 2)
        max_q = self.config.get('max_q', 5)
        seasonal = self.config.get('seasonal', True)
        m = self.config.get('m', 12)
        garch_p = self.config.get('garch_p', 1)
        garch_q = self.config.get('garch_q', 1)
        
        # Fit ARIMA model
        logger.info("Fitting ARIMA model...")
        self.arima_model = auto_arima(
            y,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q,
            seasonal=seasonal,
            m=m,
            suppress_warnings=True,
            error_action='ignore'
        )
        
        # Get ARIMA residuals
        self.residuals = self.arima_model.resid()
        
        # Fit GARCH model
        logger.info("Fitting GARCH model...")
        self.garch_model = arch_model(
            self.residuals,
            p=garch_p,
            q=garch_q,
            vol='GARCH',
            dist='normal'
        ).fit(disp='off')
        
        self.is_fitted = True
        logger.info(f"ARIMA order: {self.arima_model.order()}")
        logger.info(f"GARCH parameters: {self.garch_model.params}")
        
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Features (not used for ARIMA-GARCH)
            horizon: Prediction horizon
            
        Returns:
            Tuple of (mean predictions, volatility predictions)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get ARIMA predictions
        arima_pred = self.arima_model.predict(n_periods=horizon)
        
        # Get GARCH predictions
        garch_pred = self.garch_model.forecast(horizon=horizon)
        vol_pred = np.sqrt(garch_pred.variance.values[-horizon:])
        
        return arima_pred, vol_pred
    
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
        raise NotImplementedError("ARIMA-GARCH does not support probability predictions")
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Save ARIMA model
        self.arima_model.save(f"{path}_arima.pkl")
        
        # Save GARCH model
        self.garch_model.save(f"{path}_garch.pkl")
        
        logger.info(f"Saved model to {path}")
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        # Load ARIMA model
        self.arima_model = auto_arima.load(f"{path}_arima.pkl")
        
        # Load GARCH model
        self.garch_model = arch_model.load(f"{path}_garch.pkl")
        
        self.is_fitted = True
        logger.info(f"Loaded model from {path}")
    
    def get_order(self) -> Tuple[int, int, int]:
        """
        Get ARIMA order.
        
        Returns:
            Tuple of (p, d, q)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting order")
        
        return self.arima_model.order()
    
    def get_garch_params(self) -> pd.Series:
        """
        Get GARCH parameters.
        
        Returns:
            Series of GARCH parameters
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting GARCH parameters")
        
        return self.garch_model.params 