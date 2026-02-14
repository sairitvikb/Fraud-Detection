import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve,
    cohen_kappa_score, matthews_corrcoef, log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    def __init__(self):
        self.metrics_history = []
        self.prediction_distributions = []
    
    def evaluate_comprehensive(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_proba: Optional[np.ndarray] = None,
                              class_names: Optional[List[str]] = None) -> Dict:
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0, average='weighted')
        metrics['f1_macro'] = f1_score(y_true, y_pred, zero_division=0, average='macro')
        metrics['f1_micro'] = f1_score(y_true, y_pred, zero_division=0, average='micro')
        
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
                metrics['pr_auc'] = average_precision_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except:
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
                metrics['log_loss'] = float('inf')
        
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_positives'] = int(cm[1, 1]) if cm.shape == (2, 2) else int(cm[1, 1])
        metrics['true_negatives'] = int(cm[0, 0]) if cm.shape == (2, 2) else int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1]) if cm.shape == (2, 2) else int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0]) if cm.shape == (2, 2) else int(cm[1, 0])
        
        if cm.shape == (2, 2):
            metrics['specificity'] = metrics['true_negatives'] / max(1, metrics['true_negatives'] + metrics['false_positives'])
            metrics['sensitivity'] = metrics['recall']
            metrics['false_positive_rate'] = metrics['false_positives'] / max(1, metrics['false_positives'] + metrics['true_negatives'])
            metrics['false_negative_rate'] = metrics['false_negatives'] / max(1, metrics['false_negatives'] + metrics['true_positives'])
            metrics['positive_predictive_value'] = metrics['true_positives'] / max(1, metrics['true_positives'] + metrics['false_positives'])
            metrics['negative_predictive_value'] = metrics['true_negatives'] / max(1, metrics['true_negatives'] + metrics['false_negatives'])
        
        metrics['balanced_accuracy'] = (metrics.get('sensitivity', 0) + metrics.get('specificity', 0)) / 2
        
        if y_proba is not None:
            metrics['brier_score'] = np.mean((y_proba[:, 1] - y_true) ** 2) if y_proba.ndim > 1 else np.mean((y_proba - y_true) ** 2)
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def calculate_calibration_metrics(self, y_true: np.ndarray, y_proba: np.ndarray,
                                     n_bins: int = 10) -> Dict:
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        accuracies = []
        confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                accuracies.append(accuracy_in_bin)
                confidences.append(avg_confidence_in_bin)
        
        mce = max([abs(acc - conf) for acc, conf in zip(accuracies, confidences)] + [0])
        
        return {
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce,
            'bins': {
                'accuracies': accuracies,
                'confidences': confidences
            }
        }
    
    def calculate_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      confidence: float = 0.95) -> Dict:
        n = len(y_true)
        accuracy = accuracy_score(y_true, y_pred)
        
        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * np.sqrt((accuracy * (1 - accuracy)) / n)
        
        return {
            'accuracy': accuracy,
            'lower_bound': max(0, accuracy - margin),
            'upper_bound': min(1, accuracy + margin),
            'confidence_level': confidence
        }
    
    def calculate_lift_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                           n_bins: int = 10) -> Dict:
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        
        df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
        df = df.sort_values('y_proba', ascending=False)
        
        df['decile'] = pd.qcut(df['y_proba'], q=n_bins, labels=False, duplicates='drop')
        
        lift_data = []
        for decile in range(n_bins):
            decile_data = df[df['decile'] == decile]
            if len(decile_data) > 0:
                lift = decile_data['y_true'].mean() / df['y_true'].mean()
                lift_data.append({
                    'decile': decile,
                    'lift': lift,
                    'response_rate': decile_data['y_true'].mean(),
                    'count': len(decile_data)
                })
        
        return {'lift_curve': lift_data}
    
    def plot_comprehensive_metrics(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = range(1, len(self.metrics_history) + 1)
        
        axes[0, 0].plot(epochs, [m['accuracy'] for m in self.metrics_history], label='Accuracy', linewidth=2)
        axes[0, 0].plot(epochs, [m['f1_score'] for m in self.metrics_history], label='F1 Score', linewidth=2)
        axes[0, 0].set_title('Accuracy and F1 Score Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, [m['precision'] for m in self.metrics_history], label='Precision', linewidth=2)
        axes[0, 1].plot(epochs, [m['recall'] for m in self.metrics_history], label='Recall', linewidth=2)
        axes[0, 1].set_title('Precision and Recall Over Time', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        if self.metrics_history and 'roc_auc' in self.metrics_history[0]:
            axes[0, 2].plot(epochs, [m['roc_auc'] for m in self.metrics_history], label='ROC-AUC', linewidth=2, color='green')
            axes[0, 2].plot(epochs, [m['pr_auc'] for m in self.metrics_history], label='PR-AUC', linewidth=2, color='orange')
            axes[0, 2].set_title('AUC Metrics Over Time', fontsize=12, fontweight='bold')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('AUC')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        if self.metrics_history:
            cm = np.array(self.metrics_history[-1]['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues', cbar_kws={'label': 'Count'})
            axes[1, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Actual')
        
        if self.metrics_history and 'cohen_kappa' in self.metrics_history[0]:
            axes[1, 1].plot(epochs, [m['cohen_kappa'] for m in self.metrics_history], label='Cohen Kappa', linewidth=2, color='purple')
            axes[1, 1].plot(epochs, [m['matthews_corrcoef'] for m in self.metrics_history], label='MCC', linewidth=2, color='red')
            axes[1, 1].set_title('Agreement Metrics', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        if self.metrics_history and 'log_loss' in self.metrics_history[0]:
            axes[1, 2].plot(epochs, [m['log_loss'] for m in self.metrics_history], label='Log Loss', linewidth=2, color='darkred')
            if 'brier_score' in self.metrics_history[0]:
                axes[1, 2].plot(epochs, [m['brier_score'] for m in self.metrics_history], label='Brier Score', linewidth=2, color='darkblue')
            axes[1, 2].set_title('Probabilistic Metrics', fontsize=12, fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Score')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
