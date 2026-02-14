import numpy as np
import torch
import shap
from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class SHAPExplainer:
    def __init__(self, model, background_data: Optional[torch.Tensor] = None):
        self.model = model
        self.background_data = background_data
        self.explainer = None
        
    def create_explainer(self, 
                        background_data: Optional[torch.Tensor] = None,
                        explainer_type: str = "kernel") -> None:
        if background_data is None:
            background_data = self.background_data
        
        if background_data is None:
            raise ValueError("Background data required for SHAP explainer")
        
        if explainer_type == "kernel":
            self.explainer = shap.KernelExplainer(
                self._model_wrapper,
                background_data.cpu().numpy() if isinstance(background_data, torch.Tensor) else background_data
            )
        elif explainer_type == "deep":
            self.explainer = shap.DeepExplainer(
                self.model,
                background_data
            )
        elif explainer_type == "gradient":
            self.explainer = shap.GradientExplainer(
                self.model,
                background_data
            )
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        
        logger.info(f"SHAP {explainer_type} explainer created")
    
    def _model_wrapper(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(X_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            return probs[:, 1].cpu().numpy()
    
    def explain(self, 
               instances: torch.Tensor,
               max_evals: int = 1000,
               nsamples: int = 100) -> Dict:
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call create_explainer() first.")
        
        self.model.eval()
        
        if isinstance(self.explainer, shap.KernelExplainer):
            shap_values = self.explainer.shap_values(
                instances.cpu().numpy() if isinstance(instances, torch.Tensor) else instances,
                nsamples=nsamples
            )
        else:
            shap_values = self.explainer.shap_values(instances)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        
        return {
            'shap_values': shap_values,
            'expected_value': expected_value,
            'base_value': expected_value,
            'data': instances.cpu().numpy() if isinstance(instances, torch.Tensor) else instances
        }
    
    def get_feature_importance(self, shap_values: np.ndarray, 
                              feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        if len(shap_values.shape) > 1:
            importance = np.abs(shap_values).mean(axis=0)
        else:
            importance = np.abs(shap_values)
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'mean_shap': shap_values.mean(axis=0) if len(shap_values.shape) > 1 else shap_values
        }).sort_values('importance', ascending=False)
        
        return df
    
    def filter_false_positives(self,
                               predictions: torch.Tensor,
                               shap_values: np.ndarray,
                               threshold: float = 0.5,
                               confidence_threshold: float = 0.7) -> torch.Tensor:
        probs = torch.softmax(predictions, dim=1)
        fraud_probs = probs[:, 1]
        
        shap_confidence = 1.0 - np.abs(shap_values).mean(axis=-1) / (np.abs(shap_values).max() + 1e-8)
        
        high_confidence_mask = (fraud_probs > threshold) & (shap_confidence > confidence_threshold)
        
        filtered_predictions = predictions.clone()
        filtered_predictions[~high_confidence_mask] = torch.tensor([1.0, 0.0])
        
        false_positive_reduction = 1.0 - high_confidence_mask.float().mean().item()
        
        logger.info(f"False positive reduction: {false_positive_reduction:.2%}")
        
        return filtered_predictions, false_positive_reduction
