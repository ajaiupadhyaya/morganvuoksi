"""
ML report generator with model interpretability.
"""
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class MLReportGenerator:
    """Generator for ML model reports and interpretability analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = config.get('output_dir', 'reports')
        self.feature_names = config.get('feature_names', [])
    
    def generate_model_report(self, model, X_train: pd.DataFrame, 
                            X_test: pd.DataFrame, y_train: pd.Series,
                            y_test: pd.Series) -> Dict:
        """Generate comprehensive model report."""
        report = {
            'performance_metrics': self._calculate_performance_metrics(
                model, X_test, y_test
            ),
            'feature_importance': self._calculate_feature_importance(
                model, X_train
            ),
            'shap_values': self._calculate_shap_values(
                model, X_train, X_test
            ),
            'regime_analysis': self._analyze_regime_performance(
                model, X_test, y_test
            ),
            'prediction_confidence': self._analyze_prediction_confidence(
                model, X_test, y_test
            )
        }
        
        return report
    
    def _calculate_performance_metrics(self, model, X_test: pd.DataFrame,
                                    y_test: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics."""
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # Basic metrics
        accuracy = (predictions == y_test).mean()
        precision = (predictions[y_test == 1] == 1).mean()
        recall = (predictions[y_test == 1] == 1).mean()
        f1 = 2 * (precision * recall) / (precision + recall)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_test, probabilities
        )
        pr_auc = auc(recall_curve, precision_curve)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    
    def _calculate_feature_importance(self, model, X_train: pd.DataFrame) -> pd.DataFrame:
        """Calculate feature importance using multiple methods."""
        # Get model-specific feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            importance = None
        
        # Calculate permutation importance
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(
            model, X_train, y_train, n_repeats=10, random_state=42
        )
        
        # Combine results
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance if importance is not None else perm_importance.importances_mean,
            'permutation_importance': perm_importance.importances_mean,
            'permutation_std': perm_importance.importances_std
        })
        
        return feature_importance.sort_values('importance', ascending=False)
    
    def _calculate_shap_values(self, model, X_train: pd.DataFrame,
                             X_test: pd.DataFrame) -> Dict:
        """Calculate SHAP values for model interpretability."""
        # Initialize SHAP explainer
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X_train)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test)
        
        # Calculate feature importance from SHAP
        if isinstance(shap_values, list):
            shap_values = np.abs(shap_values[1])
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        })
        
        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance.sort_values(
                'shap_importance', ascending=False
            )
        }
    
    def _analyze_regime_performance(self, model, X_test: pd.DataFrame,
                                  y_test: pd.Series) -> Dict:
        """Analyze model performance across different market regimes."""
        if 'Regime' not in X_test.columns:
            return {}
        
        regime_performance = {}
        for regime in X_test['Regime'].unique():
            mask = X_test['Regime'] == regime
            regime_metrics = self._calculate_performance_metrics(
                model, X_test[mask], y_test[mask]
            )
            regime_performance[regime] = regime_metrics
        
        return regime_performance
    
    def _analyze_prediction_confidence(self, model, X_test: pd.DataFrame,
                                     y_test: pd.Series) -> Dict:
        """Analyze prediction confidence and calibration."""
        if not hasattr(model, 'predict_proba'):
            return {}
        
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # Calculate calibration curve
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(
            y_test, probabilities, n_bins=10
        )
        
        # Calculate confidence intervals
        confidence_intervals = []
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            mask = probabilities >= threshold
            if mask.any():
                accuracy = (predictions[mask] == y_test[mask]).mean()
                confidence_intervals.append({
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'coverage': mask.mean()
                })
        
        return {
            'calibration_curve': {
                'prob_true': prob_true,
                'prob_pred': prob_pred
            },
            'confidence_intervals': confidence_intervals
        }
    
    def plot_feature_importance(self, importance: pd.DataFrame,
                              save_path: Optional[str] = None):
        """Plot feature importance."""
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_shap_summary(self, shap_values: np.ndarray,
                         X_test: pd.DataFrame, save_path: Optional[str] = None):
        """Plot SHAP summary."""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=self.feature_names)
        plt.title('SHAP Summary Plot')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_regime_performance(self, regime_performance: Dict,
                              save_path: Optional[str] = None):
        """Plot performance across regimes."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        regimes = list(regime_performance.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [regime_performance[r][metric] for r in regimes]
            sns.barplot(x=regimes, y=values, ax=axes[i])
            axes[i].set_title(f'{metric.title()} by Regime')
            axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_prediction_confidence(self, confidence_analysis: Dict,
                                 save_path: Optional[str] = None):
        """Plot prediction confidence analysis."""
        if not confidence_analysis:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Calibration curve
        prob_true = confidence_analysis['calibration_curve']['prob_true']
        prob_pred = confidence_analysis['calibration_curve']['prob_pred']
        ax1.plot(prob_pred, prob_true, 'b-', label='Model')
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('True Probability')
        ax1.set_title('Calibration Curve')
        ax1.legend()
        
        # Confidence intervals
        intervals = confidence_analysis['confidence_intervals']
        thresholds = [i['threshold'] for i in intervals]
        accuracies = [i['accuracy'] for i in intervals]
        coverages = [i['coverage'] for i in intervals]
        
        ax2.plot(thresholds, accuracies, 'b-', label='Accuracy')
        ax2.plot(thresholds, coverages, 'r--', label='Coverage')
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('Value')
        ax2.set_title('Confidence Analysis')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def generate_report(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_train: pd.Series, y_test: pd.Series,
                       output_dir: Optional[str] = None) -> None:
        """Generate and save complete ML report."""
        if output_dir is None:
            output_dir = self.output_dir
        
        # Generate report data
        report = self.generate_model_report(model, X_train, X_test, y_train, y_test)
        
        # Create plots
        self.plot_feature_importance(
            report['feature_importance'],
            f'{output_dir}/feature_importance.png'
        )
        
        self.plot_shap_summary(
            report['shap_values']['shap_values'],
            X_test,
            f'{output_dir}/shap_summary.png'
        )
        
        if report['regime_analysis']:
            self.plot_regime_performance(
                report['regime_analysis'],
                f'{output_dir}/regime_performance.png'
            )
        
        self.plot_prediction_confidence(
            report['prediction_confidence'],
            f'{output_dir}/prediction_confidence.png'
        )
        
        # Save metrics
        pd.DataFrame(report['performance_metrics'], index=[0]).to_csv(
            f'{output_dir}/performance_metrics.csv'
        )
        
        report['feature_importance'].to_csv(
            f'{output_dir}/feature_importance.csv'
        )
        
        if report['regime_analysis']:
            pd.DataFrame(report['regime_analysis']).to_csv(
                f'{output_dir}/regime_performance.csv'
            )
        
        logger.info(f"ML report generated and saved to {output_dir}") 