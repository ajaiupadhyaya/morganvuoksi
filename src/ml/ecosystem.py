"""
Advanced ML model ecosystem for quantitative trading.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoModel, AutoTokenizer
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.models import NBeats
from pytorch_forecasting.models.deepar import DeepAR
from pytorch_forecasting.models.wavenet import WaveNet
from pytorch_forecasting.models.lstm import LSTNet
from pytorch_forecasting.models.informer import Informer
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class MLEcosystem:
    """Advanced ML model ecosystem."""
    
    def __init__(self, config: Dict):
        self.config = config
        self._setup_models()
        self._setup_tokenizers()
    
    def _setup_models(self):
        """Setup ML models."""
        # Financial LLMs
        if 'finbert' in self.config:
            self.finbert = AutoModel.from_pretrained("ProsusAI/finbert")
        
        if 'bloomberggpt' in self.config:
            self.bloomberggpt = AutoModel.from_pretrained("bloomberg/bloomberg-gpt")
        
        # Time Series Models
        self.tft = TemporalFusionTransformer.from_dataset(
            self.config.get('tft_dataset'),
            learning_rate=0.03,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            loss=torch.nn.MSELoss(),
            log_interval=10,
            reduce_on_plateau_patience=4
        )
        
        self.nbeats = NBeats.from_dataset(
            self.config.get('nbeats_dataset'),
            learning_rate=0.001,
            log_interval=10,
            log_val_interval=1,
            weight_decay=1e-2,
            widths=[32, 512],
            backcast_loss_ratio=1.0
        )
        
        self.deepar = DeepAR.from_dataset(
            self.config.get('deepar_dataset'),
            learning_rate=0.001,
            log_interval=10,
            log_val_interval=1,
            hidden_size=32,
            rnn_layers=2
        )
        
        self.wavenet = WaveNet.from_dataset(
            self.config.get('wavenet_dataset'),
            learning_rate=0.001,
            log_interval=10,
            log_val_interval=1,
            hidden_size=32,
            num_filters=32,
            num_layers=4
        )
        
        self.lstnet = LSTNet.from_dataset(
            self.config.get('lstnet_dataset'),
            learning_rate=0.001,
            log_interval=10,
            log_val_interval=1,
            hidden_size=32,
            num_layers=2
        )
        
        self.informer = Informer.from_dataset(
            self.config.get('informer_dataset'),
            learning_rate=0.001,
            log_interval=10,
            log_val_interval=1,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )
    
    def _setup_tokenizers(self):
        """Setup tokenizers for LLMs."""
        if 'finbert' in self.config:
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        
        if 'bloomberggpt' in self.config:
            self.bloomberggpt_tokenizer = AutoTokenizer.from_pretrained("bloomberg/bloomberg-gpt")
    
    def analyze_sentiment(self, text: str, model: str = 'finbert') -> Dict:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            model: Model to use ('finbert' or 'bloomberggpt')
            
        Returns:
            Sentiment analysis results
        """
        try:
            if model == 'finbert':
                tokenizer = self.finbert_tokenizer
                model = self.finbert
            else:
                tokenizer = self.bloomberggpt_tokenizer
                model = self.bloomberggpt
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
            
            # Get sentiment
            sentiment = {
                'positive': predictions[0][0].item(),
                'negative': predictions[0][1].item(),
                'neutral': predictions[0][2].item()
            }
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {}
    
    def forecast_time_series(self, data: pd.DataFrame,
                           model: str = 'tft',
                           horizon: int = 10) -> pd.DataFrame:
        """
        Forecast time series.
        
        Args:
            data: Time series data
            model: Model to use
            horizon: Forecast horizon
            
        Returns:
            Forecast results
        """
        try:
            if model == 'tft':
                model = self.tft
            elif model == 'nbeats':
                model = self.nbeats
            elif model == 'deepar':
                model = self.deepar
            elif model == 'wavenet':
                model = self.wavenet
            elif model == 'lstnet':
                model = self.lstnet
            elif model == 'informer':
                model = self.informer
            else:
                raise ValueError(f"Unknown model: {model}")
            
            # Prepare data
            dataset = self._prepare_dataset(data, model)
            
            # Make predictions
            predictions = model.predict(dataset)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error forecasting time series: {str(e)}")
            return pd.DataFrame()
    
    def _prepare_dataset(self, data: pd.DataFrame,
                        model: nn.Module) -> torch.utils.data.Dataset:
        """
        Prepare dataset for model.
        
        Args:
            data: Time series data
            model: Model to prepare data for
            
        Returns:
            Prepared dataset
        """
        # Convert to PyTorch tensors
        if isinstance(data, pd.DataFrame):
            data = torch.tensor(data.values, dtype=torch.float32)
        
        # Create dataset
        if isinstance(model, TemporalFusionTransformer):
            return self._prepare_tft_dataset(data)
        elif isinstance(model, NBeats):
            return self._prepare_nbeats_dataset(data)
        elif isinstance(model, DeepAR):
            return self._prepare_deepar_dataset(data)
        elif isinstance(model, WaveNet):
            return self._prepare_wavenet_dataset(data)
        elif isinstance(model, LSTNet):
            return self._prepare_lstnet_dataset(data)
        elif isinstance(model, Informer):
            return self._prepare_informer_dataset(data)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
    
    def _prepare_tft_dataset(self, data: torch.Tensor) -> torch.utils.data.Dataset:
        """Prepare dataset for TFT."""
        # Implementation depends on your data structure
        pass
    
    def _prepare_nbeats_dataset(self, data: torch.Tensor) -> torch.utils.data.Dataset:
        """Prepare dataset for N-BEATS."""
        # Implementation depends on your data structure
        pass
    
    def _prepare_deepar_dataset(self, data: torch.Tensor) -> torch.utils.data.Dataset:
        """Prepare dataset for DeepAR."""
        # Implementation depends on your data structure
        pass
    
    def _prepare_wavenet_dataset(self, data: torch.Tensor) -> torch.utils.data.Dataset:
        """Prepare dataset for WaveNet."""
        # Implementation depends on your data structure
        pass
    
    def _prepare_lstnet_dataset(self, data: torch.Tensor) -> torch.utils.data.Dataset:
        """Prepare dataset for LSTNet."""
        # Implementation depends on your data structure
        pass
    
    def _prepare_informer_dataset(self, data: torch.Tensor) -> torch.utils.data.Dataset:
        """Prepare dataset for Informer."""
        # Implementation depends on your data structure
        pass
    
    def save_models(self, path: str):
        """
        Save all models.
        
        Args:
            path: Path to save models
        """
        try:
            # Save time series models
            torch.save(self.tft.state_dict(), f"{path}/tft.pt")
            torch.save(self.nbeats.state_dict(), f"{path}/nbeats.pt")
            torch.save(self.deepar.state_dict(), f"{path}/deepar.pt")
            torch.save(self.wavenet.state_dict(), f"{path}/wavenet.pt")
            torch.save(self.lstnet.state_dict(), f"{path}/lstnet.pt")
            torch.save(self.informer.state_dict(), f"{path}/informer.pt")
            
            # Save LLMs
            if hasattr(self, 'finbert'):
                self.finbert.save_pretrained(f"{path}/finbert")
            if hasattr(self, 'bloomberggpt'):
                self.bloomberggpt.save_pretrained(f"{path}/bloomberggpt")
            
            logger.info(f"Saved models to {path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self, path: str):
        """
        Load all models.
        
        Args:
            path: Path to load models from
        """
        try:
            # Load time series models
            self.tft.load_state_dict(torch.load(f"{path}/tft.pt"))
            self.nbeats.load_state_dict(torch.load(f"{path}/nbeats.pt"))
            self.deepar.load_state_dict(torch.load(f"{path}/deepar.pt"))
            self.wavenet.load_state_dict(torch.load(f"{path}/wavenet.pt"))
            self.lstnet.load_state_dict(torch.load(f"{path}/lstnet.pt"))
            self.informer.load_state_dict(torch.load(f"{path}/informer.pt"))
            
            # Load LLMs
            if 'finbert' in self.config:
                self.finbert = AutoModel.from_pretrained(f"{path}/finbert")
            if 'bloomberggpt' in self.config:
                self.bloomberggpt = AutoModel.from_pretrained(f"{path}/bloomberggpt")
            
            logger.info(f"Loaded models from {path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}") 