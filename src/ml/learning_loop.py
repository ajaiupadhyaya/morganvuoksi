"""
ML self-learning engine for adaptive quantitative trading.
Implements periodic retraining, signal quality tracking, and regime switching.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .safety import ModelSafetyMonitor, CrossValidator, PositionSizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for managing ML models and their metadata."""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.models = {}
        self.metadata = {}
    
    def save_model(self, model_id: str, model: object, metadata: Dict):
        """Save model and its metadata."""
        try:
            # Save model
            if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
                joblib.dump(model, self.model_dir / f"{model_id}.joblib")
            elif isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier)):
                model.save_model(str(self.model_dir / f"{model_id}.json"))
            elif isinstance(model, Sequential):
                model.save(self.model_dir / f"{model_id}.h5")
            elif isinstance(model, nn.Module):
                torch.save(model.state_dict(), self.model_dir / f"{model_id}.pt")
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")
            
            # Save metadata
            self.metadata[model_id] = {
                **metadata,
                'last_updated': datetime.now().isoformat()
            }
            joblib.dump(self.metadata, self.model_dir / 'metadata.joblib')
            
            logger.info(f"Saved model {model_id}")
            
        except Exception as e:
            logger.error(f"Error saving model {model_id}: {str(e)}")
            raise
    
    def load_model(self, model_id: str) -> Tuple[object, Dict]:
        """Load model and its metadata."""
        try:
            # Load metadata
            metadata = self.metadata.get(model_id)
            if not metadata:
                raise ValueError(f"Model {model_id} not found in registry")
            
            # Load model
            model_path = self.model_dir / model_id
            if model_path.with_suffix('.joblib').exists():
                model = joblib.load(model_path.with_suffix('.joblib'))
            elif model_path.with_suffix('.json').exists():
                if 'xgb' in model_id:
                    model = xgb.XGBClassifier()
                    model.load_model(str(model_path.with_suffix('.json')))
                else:
                    model = lgb.LGBMClassifier()
                    model.load_model(str(model_path.with_suffix('.json')))
            elif model_path.with_suffix('.h5').exists():
                model = load_model(model_path.with_suffix('.h5'))
            elif model_path.with_suffix('.pt').exists():
                # Load PyTorch model
                model_class = self._get_model_class(model_id)
                model = model_class()
                model.load_state_dict(torch.load(model_path.with_suffix('.pt')))
            else:
                raise ValueError(f"No model file found for {model_id}")
            
            logger.info(f"Loaded model {model_id}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            raise
    
    def _get_model_class(self, model_id: str) -> type:
        """Get model class based on model ID."""
        # Map model IDs to their classes
        model_classes = {
            'lstm': LSTMClassifier,
            'transformer': TransformerClassifier
        }
        return model_classes.get(model_id.split('_')[0])

class SignalQualityTracker:
    """Tracks signal quality metrics over time."""
    
    def __init__(self, window: int = 252):
        self.window = window
        self.metrics_history = []
    
    def update(self, signals: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """Update signal quality metrics."""
        try:
            # Calculate metrics
            precision = precision_score(returns > 0, signals > 0)
            auc = roc_auc_score(returns > 0, signals)
            
            # Calculate forward return correlation
            forward_returns = returns.shift(-1)
            correlation = signals.corr(forward_returns)
            
            # Calculate signal decay
            decay_windows = [1, 5, 10, 20]
            decay_metrics = {}
            for window in decay_windows:
                decay_returns = returns.shift(-window)
                decay_metrics[f'corr_{window}d'] = signals.corr(decay_returns)
            
            metrics = {
                'precision': precision,
                'auc': auc,
                'correlation': correlation,
                **decay_metrics,
                'timestamp': datetime.now()
            }
            
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.window:
                self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error updating signal quality metrics: {str(e)}")
            return {}
    
    def get_metrics(self) -> pd.DataFrame:
        """Get historical metrics."""
        return pd.DataFrame(self.metrics_history)

class RegimeDetector:
    """Detects market regimes and adjusts model weights accordingly."""
    
    def __init__(self, window: int = 60):
        self.window = window
        self.regime_history = []
    
    def detect_regime(self, returns: pd.Series) -> str:
        """Detect current market regime."""
        try:
            # Calculate regime indicators
            volatility = returns.rolling(self.window).std()
            trend = returns.rolling(self.window).mean()
            momentum = returns.rolling(self.window).sum()
            
            # Determine regime
            if volatility.iloc[-1] > volatility.quantile(0.8):
                regime = 'high_volatility'
            elif trend.iloc[-1] > 0 and momentum.iloc[-1] > 0:
                regime = 'bullish'
            elif trend.iloc[-1] < 0 and momentum.iloc[-1] < 0:
                regime = 'bearish'
            else:
                regime = 'neutral'
            
            self.regime_history.append({
                'regime': regime,
                'timestamp': datetime.now()
            })
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting regime: {str(e)}")
            return 'unknown'
    
    def get_regime_weights(self) -> Dict[str, float]:
        """Get model weights based on current regime."""
        try:
            current_regime = self.regime_history[-1]['regime']
            
            # Define regime-specific weights
            weights = {
                'high_volatility': {
                    'xgb': 0.3,
                    'lstm': 0.4,
                    'transformer': 0.3
                },
                'bullish': {
                    'xgb': 0.4,
                    'lstm': 0.3,
                    'transformer': 0.3
                },
                'bearish': {
                    'xgb': 0.3,
                    'lstm': 0.3,
                    'transformer': 0.4
                },
                'neutral': {
                    'xgb': 0.33,
                    'lstm': 0.33,
                    'transformer': 0.34
                }
            }
            
            return weights.get(current_regime, weights['neutral'])
            
        except Exception as e:
            logger.error(f"Error getting regime weights: {str(e)}")
            return {'xgb': 0.33, 'lstm': 0.33, 'transformer': 0.34}

class LearningLoop:
    """Main learning loop for model retraining and adaptation."""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        signal_tracker: SignalQualityTracker,
        regime_detector: RegimeDetector,
        retrain_interval: int = 5,  # days
        min_samples: int = 1000
    ):
        self.model_registry = model_registry
        self.signal_tracker = signal_tracker
        self.regime_detector = regime_detector
        self.retrain_interval = retrain_interval
        self.min_samples = min_samples
        self.last_retrain = None
        
        # Initialize safety components
        self.safety_monitor = ModelSafetyMonitor()
        self.cross_validator = CrossValidator()
        self.position_sizer = PositionSizer()
    
    def should_retrain(self) -> bool:
        """Check if models should be retrained."""
        if not self.last_retrain:
            return True
        
        days_since_retrain = (datetime.now() - self.last_retrain).days
        return days_since_retrain >= self.retrain_interval
    
    def retrain_models(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, object]:
        """Retrain all models with latest data."""
        try:
            if len(features) < self.min_samples:
                logger.warning(f"Insufficient samples for retraining: {len(features)} < {self.min_samples}")
                return {}
            
            # Detect current regime
            regime = self.regime_detector.detect_regime(returns)
            logger.info(f"Current regime: {regime}")
            
            # Train models
            models = {}
            
            # XGBoost
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1
            )
            # Cross-validate
            xgb_score, xgb_metrics = self.cross_validator.validate(
                xgb_model,
                features,
                returns > 0
            )
            if xgb_score > 0.5:  # Only use if performance is acceptable
                xgb_model.fit(features, returns > 0)
                models['xgb'] = xgb_model
            
            # LSTM
            lstm_model = LSTMClassifier(
                input_dim=features.shape[1],
                hidden_dim=64,
                output_dim=1
            )
            # Cross-validate with early stopping
            lstm_score, lstm_metrics = self.cross_validator.validate(
                lstm_model,
                features,
                returns > 0,
                is_deep_learning=True
            )
            if lstm_score > 0.5:
                lstm_model.fit(features, returns > 0)
                models['lstm'] = lstm_model
            
            # Transformer
            transformer_model = TransformerClassifier(
                input_dim=features.shape[1],
                hidden_dim=128,
                num_heads=4
            )
            # Cross-validate with early stopping
            transformer_score, transformer_metrics = self.cross_validator.validate(
                transformer_model,
                features,
                returns > 0,
                is_deep_learning=True
            )
            if transformer_score > 0.5:
                transformer_model.fit(features, returns > 0)
                models['transformer'] = transformer_model
            
            # Save models with metadata
            for model_id, model in models.items():
                metrics = {
                    'regime': regime,
                    'n_samples': len(features),
                    'feature_importance': self._get_feature_importance(model, features),
                    'cross_val_score': {
                        'xgb': xgb_score,
                        'lstm': lstm_score,
                        'transformer': transformer_score
                    }[model_id]
                }
                self.model_registry.save_model(model_id, model, metrics)
            
            self.last_retrain = datetime.now()
            logger.info("Models retrained successfully")
            
            return models
            
        except Exception as e:
            logger.error(f"Error retraining models: {str(e)}")
            return {}
    
    def _get_feature_importance(self, model: object, features: pd.DataFrame) -> pd.Series:
        """Get feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                return pd.Series(
                    model.feature_importances_,
                    index=features.columns
                )
            elif hasattr(model, 'coef_'):
                return pd.Series(
                    np.abs(model.coef_[0]),
                    index=features.columns
                )
            else:
                return pd.Series(0, index=features.columns)
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return pd.Series(0, index=features.columns)
    
    def generate_signals(
        self,
        features: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """Generate trading signals using ensemble of models."""
        try:
            # Get regime-specific weights
            weights = self.regime_detector.get_regime_weights()
            
            # Load models and generate predictions
            predictions = {}
            confidences = {}
            for model_id, weight in weights.items():
                # Skip if circuit breaker is triggered
                if self.safety_monitor.is_circuit_breaker_triggered(model_id):
                    logger.warning(f"Skipping model {model_id} due to circuit breaker")
                    continue
                
                model, metadata = self.model_registry.load_model(model_id)
                pred = model.predict_proba(features)[:, 1]
                
                # Check model performance
                if not self.safety_monitor.check_performance(model_id, metadata):
                    logger.warning(f"Model {model_id} failed performance check")
                    continue
                
                # Check prediction confidence
                if not self.safety_monitor.check_confidence(model_id, pred):
                    logger.warning(f"Model {model_id} failed confidence check")
                    continue
                
                predictions[model_id] = pred * weight
                confidences[model_id] = np.mean(pred)
            
            # Combine predictions
            ensemble_signal = pd.Series(0, index=features.index)
            for pred in predictions.values():
                ensemble_signal += pred
            
            # Check signal volatility
            for model_id in predictions.keys():
                if not self.safety_monitor.check_signal_volatility(
                    model_id,
                    pd.Series(predictions[model_id])
                ):
                    logger.warning(f"Model {model_id} failed volatility check")
            
            # Calculate position sizes and stop-losses
            position_sizes = {}
            stop_losses = {}
            for model_id, signal in predictions.items():
                confidence = confidences[model_id]
                volatility = market_data['returns'].std() if market_data is not None else None
                
                position_sizes[model_id] = self.position_sizer.calculate_position_size(
                    signal[-1],  # Use latest signal
                    confidence,
                    volatility
                )
                
                stop_losses[model_id] = self.position_sizer.calculate_stop_loss(
                    position_sizes[model_id],
                    volatility if volatility is not None else 0.02,
                    confidence
                )
            
            # Update signal quality metrics
            if market_data is not None:
                self.signal_tracker.update(ensemble_signal, market_data['returns'])
            
            return ensemble_signal, {
                'position_sizes': position_sizes,
                'stop_losses': stop_losses,
                'confidences': confidences
            }
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return pd.Series(0, index=features.index), {}

class LSTMClassifier(nn.Module):
    """LSTM-based classifier for time series prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(out)

class TransformerClassifier(nn.Module):
    """Transformer-based classifier for time series prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads
            ),
            num_layers=2
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x) 