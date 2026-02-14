import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class AdvancedEvaluator:
    def __init__(self):
        self.metrics_history = []
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                y_proba: Optional[np.ndarray] = None) -> Dict:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                metrics['pr_auc'] = average_precision_score(y_true, y_proba)
            except:
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_positives'] = int(cm[1, 1])
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        
        metrics['specificity'] = metrics['true_negatives'] / max(1, metrics['true_negatives'] + metrics['false_positives'])
        metrics['sensitivity'] = metrics['recall']
        metrics['false_positive_rate'] = metrics['false_positives'] / max(1, metrics['false_positives'] + metrics['true_negatives'])
        metrics['false_negative_rate'] = metrics['false_negatives'] / max(1, metrics['false_negatives'] + metrics['true_positives'])
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def calculate_false_positive_reduction(self, baseline_fp: int, current_fp: int) -> float:
        reduction = (baseline_fp - current_fp) / max(1, baseline_fp)
        return reduction
    
    def plot_metrics(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = range(1, len(self.metrics_history) + 1)
        
        axes[0, 0].plot(epochs, [m['accuracy'] for m in self.metrics_history], label='Accuracy')
        axes[0, 0].set_title('Accuracy Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(epochs, [m['precision'] for m in self.metrics_history], label='Precision')
        axes[0, 1].plot(epochs, [m['recall'] for m in self.metrics_history], label='Recall')
        axes[0, 1].plot(epochs, [m['f1_score'] for m in self.metrics_history], label='F1')
        axes[0, 1].set_title('Precision, Recall, F1 Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        if self.metrics_history and 'roc_auc' in self.metrics_history[0]:
            axes[1, 0].plot(epochs, [m['roc_auc'] for m in self.metrics_history], label='ROC-AUC')
            axes[1, 0].plot(epochs, [m['pr_auc'] for m in self.metrics_history], label='PR-AUC')
            axes[1, 0].set_title('AUC Metrics Over Time')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        if self.metrics_history:
            cm = np.array(self.metrics_history[-1]['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 1], cmap='Blues')
            axes[1, 1].set_title('Confusion Matrix')
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
