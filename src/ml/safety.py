"""
Model safety and risk control module.
Implements cross-validation, early stopping, performance monitoring, and risk controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_score
import torch
import torch.nn as nn
try:  # Optional TensorFlow dependency
    from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
except Exception:  # pragma: no cover - TensorFlow not installed
    EarlyStopping = None
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_safety.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelSafetyMonitor:
    """Monitors model safety and performance."""
    
    def __init__(
        self,
        performance_window: int = 252,  # 1 year of trading days
        min_performance: float = 0.5,  # Minimum AUC score
        max_drawdown: float = 0.1,  # Maximum drawdown allowed
        confidence_threshold: float = 0.7,  # Minimum confidence threshold
        volatility_threshold: float = 0.02  # Maximum signal volatility
    ):
        self.performance_window = performance_window
        self.min_performance = min_performance
        self.max_drawdown = max_drawdown
        self.confidence_threshold = confidence_threshold
        self.volatility_threshold = volatility_threshold
        
        self.performance_history = []
        self.warnings = []
        self.circuit_breakers = set()
    
    def check_performance(self, model_id: str, metrics: Dict[str, float]) -> bool:
        """Check if model performance meets requirements."""
        try:
            # Add metrics to history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'model_id': model_id,
                **metrics
            })
            
            # Keep only recent history
            if len(self.performance_history) > self.performance_window:
                self.performance_history.pop(0)
            
            # Calculate performance metrics
            recent_metrics = pd.DataFrame(self.performance_history)
            recent_metrics = recent_metrics[recent_metrics['model_id'] == model_id]
            
            if len(recent_metrics) < 10:  # Need minimum samples
                return True
            
            # Check AUC
            if recent_metrics['auc'].mean() < self.min_performance:
                self._add_warning(
                    model_id,
                    'performance',
                    f"AUC below threshold: {recent_metrics['auc'].mean():.3f}"
                )
                return False
            
            # Check drawdown
            drawdown = self._calculate_drawdown(recent_metrics['auc'])
            if drawdown > self.max_drawdown:
                self._add_warning(
                    model_id,
                    'drawdown',
                    f"Drawdown too high: {drawdown:.3f}"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking performance: {str(e)}")
            return False
    
    def check_confidence(self, model_id: str, predictions: np.ndarray) -> bool:
        """Check if model predictions meet confidence requirements."""
        try:
            # Calculate prediction entropy
            entropy = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)
            mean_entropy = np.mean(entropy)
            
            if mean_entropy > -np.log(self.confidence_threshold):
                self._add_warning(
                    model_id,
                    'confidence',
                    f"Low confidence: {mean_entropy:.3f}"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking confidence: {str(e)}")
            return False
    
    def check_signal_volatility(
        self,
        model_id: str,
        signals: pd.Series,
        window: int = 20
    ) -> bool:
        """Check if signal volatility is within acceptable range."""
        try:
            # Calculate rolling volatility
            volatility = signals.rolling(window).std()
            recent_volatility = volatility.iloc[-1]
            
            if recent_volatility > self.volatility_threshold:
                self._add_warning(
                    model_id,
                    'volatility',
                    f"High signal volatility: {recent_volatility:.3f}"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking signal volatility: {str(e)}")
            return False
    
    def _calculate_drawdown(self, series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        rolling_max = series.expanding().max()
        drawdown = (series - rolling_max) / rolling_max
        return abs(drawdown.min())
    
    def _add_warning(
        self,
        model_id: str,
        warning_type: str,
        message: str
    ):
        """Add warning and potentially trigger circuit breaker."""
        warning = {
            'timestamp': datetime.now(),
            'model_id': model_id,
            'type': warning_type,
            'message': message
        }
        self.warnings.append(warning)
        logger.warning(f"Model {model_id}: {message}")
        
        # Check if circuit breaker should be triggered
        recent_warnings = [
            w for w in self.warnings
            if w['model_id'] == model_id
            and (datetime.now() - w['timestamp']).days < 5
        ]
        
        if len(recent_warnings) >= 3:
            self.circuit_breakers.add(model_id)
            logger.error(f"Circuit breaker triggered for model {model_id}")
    
    def get_warnings(self, model_id: Optional[str] = None) -> List[Dict]:
        """Get recent warnings."""
        if model_id:
            return [
                w for w in self.warnings
                if w['model_id'] == model_id
            ]
        return self.warnings
    
    def is_circuit_breaker_triggered(self, model_id: str) -> bool:
        """Check if circuit breaker is triggered for model."""
        return model_id in self.circuit_breakers
    
    def reset_circuit_breaker(self, model_id: str):
        """Reset circuit breaker for model."""
        self.circuit_breakers.discard(model_id)
        logger.info(f"Circuit breaker reset for model {model_id}")

class CrossValidator:
    """Implements k-fold cross-validation with early stopping."""
    
    def __init__(
        self,
        n_splits: int = 5,
        early_stopping_rounds: int = 10,
        min_delta: float = 0.001
    ):
        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.min_delta = min_delta
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    def validate(
        self,
        model: Union[nn.Module, object],
        features: pd.DataFrame,
        targets: pd.Series,
        is_deep_learning: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """Perform k-fold cross-validation."""
        try:
            scores = []
            feature_importance = []
            
            for fold, (train_idx, val_idx) in enumerate(self.kf.split(features)):
                X_train = features.iloc[train_idx]
                y_train = targets.iloc[train_idx]
                X_val = features.iloc[val_idx]
                y_val = targets.iloc[val_idx]
                
                if is_deep_learning:
                    # Deep learning with optional early stopping
                    model_copy = self._clone_model(model)
                    callbacks = []
                    if EarlyStopping is not None:
                        callbacks.append(
                            EarlyStopping(
                                monitor='val_loss',
                                patience=self.early_stopping_rounds,
                                min_delta=self.min_delta,
                                restore_best_weights=True,
                            )
                        )

                    model_copy.fit(
                        X_train,
                        y_train,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks,
                    )
                    
                    # Get predictions
                    val_pred = model_copy.predict(X_val)
                    
                else:
                    # Traditional ML
                    model_copy = self._clone_model(model)
                    model_copy.fit(X_train, y_train)
                    val_pred = model_copy.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                fold_score = roc_auc_score(y_val, val_pred)
                scores.append(fold_score)
                
                # Get feature importance if available
                if hasattr(model_copy, 'feature_importances_'):
                    feature_importance.append(
                        pd.Series(
                            model_copy.feature_importances_,
                            index=features.columns
                        )
                    )
            
            # Calculate average metrics
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Calculate average feature importance
            if feature_importance:
                avg_importance = pd.concat(feature_importance).mean()
            else:
                avg_importance = pd.Series(0, index=features.columns)
            
            return mean_score, {
                'mean_auc': mean_score,
                'std_auc': std_score,
                'feature_importance': avg_importance
            }
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            return 0.0, {}
    
    def _clone_model(self, model: Union[nn.Module, object]) -> Union[nn.Module, object]:
        """Create a copy of the model."""
        if isinstance(model, nn.Module):
            return type(model)(**model.__dict__)
        else:
            return type(model)(**model.get_params())

class PositionSizer:
    """Determines position sizes based on model confidence and risk limits."""
    
    def __init__(
        self,
        max_position_size: float = 0.1,  # Maximum position size as fraction of portfolio
        min_confidence: float = 0.7,  # Minimum confidence for full position
        volatility_scaling: bool = True  # Whether to scale by volatility
    ):
        self.max_position_size = max_position_size
        self.min_confidence = min_confidence
        self.volatility_scaling = volatility_scaling
    
    def calculate_position_size(
        self,
        signal: float,
        confidence: float,
        volatility: Optional[float] = None
    ) -> float:
        """Calculate position size based on signal, confidence, and volatility."""
        try:
            # Base position size from signal
            base_size = abs(signal) * self.max_position_size
            
            # Scale by confidence
            confidence_scale = min(confidence / self.min_confidence, 1.0)
            position_size = base_size * confidence_scale
            
            # Scale by volatility if enabled
            if self.volatility_scaling and volatility is not None:
                vol_scale = 1.0 / (1.0 + volatility)
                position_size *= vol_scale
            
            # Ensure position size is within limits
            position_size = min(position_size, self.max_position_size)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    def calculate_stop_loss(
        self,
        position_size: float,
        volatility: float,
        confidence: float
    ) -> float:
        """Calculate stop-loss level based on position size and risk metrics."""
        try:
            # Base stop-loss on volatility
            base_stop = volatility * 2.0
            
            # Adjust by confidence
            confidence_scale = max(1.0 - confidence, 0.5)
            stop_loss = base_stop * confidence_scale
            
            # Adjust by position size
            size_scale = min(position_size / self.max_position_size, 1.0)
            stop_loss *= size_scale
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating stop-loss: {str(e)}")
            return 0.0 
