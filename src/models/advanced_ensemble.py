import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
import logging
from .vision_transformer import VisionTransformerFraudDetector, MultiScaleVisionTransformer
from .graph_neural_network import GATFraudDetector, GCNFraudDetector, GraphSAGEFraudDetector, GINFraudDetector, TemporalGNN

logger = logging.getLogger(__name__)


class AdaptiveWeightedEnsemble(nn.Module):
    def __init__(self,
                 models: List[nn.Module],
                 num_classes: int = 2,
                 use_meta_learner: bool = True,
                 use_attention_fusion: bool = True):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
        self.use_meta_learner = use_meta_learner
        self.use_attention_fusion = use_attention_fusion
        
        if use_attention_fusion:
            self.attention_fusion = nn.MultiheadAttention(
                embed_dim=num_classes,
                num_heads=4,
                batch_first=True
            )
            self.fusion_proj = nn.Linear(num_classes * len(models), num_classes)
        else:
            self.fusion_proj = nn.Linear(num_classes * len(models), num_classes)
        
        if use_meta_learner:
            self.meta_learner = nn.Sequential(
                nn.Linear(num_classes * len(models) + len(models), 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
        
        self.adaptive_weights = nn.Parameter(torch.ones(len(models)) / len(models))
        self.confidence_estimator = nn.Sequential(
            nn.Linear(num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, inputs: Dict) -> Tuple[torch.Tensor, Dict]:
        model_outputs = []
        confidences = []
        
        for i, model in enumerate(self.models):
            if 'pixel_values' in inputs and hasattr(model, 'extract_features'):
                output = model(inputs['pixel_values'])
            elif 'x' in inputs and 'edge_index' in inputs:
                output, _ = model(inputs['x'], inputs['edge_index'], inputs.get('batch'))
            else:
                output = model(inputs.get('data', inputs))
            
            if isinstance(output, tuple):
                output = output[0]
            
            probs = F.softmax(output, dim=1)
            model_outputs.append(probs)
            
            confidence = self.confidence_estimator(output)
            confidences.append(confidence.squeeze())
        
        model_outputs = torch.stack(model_outputs, dim=1)
        confidences = torch.stack(confidences, dim=1)
        
        if self.use_attention_fusion:
            attn_output, attn_weights = self.attention_fusion(
                model_outputs, model_outputs, model_outputs
            )
            fused = attn_output.mean(dim=1)
            stacked = torch.cat([fused] + [m.mean(dim=0, keepdim=True) for m in model_outputs], dim=1)
        else:
            stacked = model_outputs.view(model_outputs.size(0), -1)
        
        if self.use_meta_learner:
            meta_input = torch.cat([stacked, confidences.mean(dim=1, keepdim=True)], dim=1)
            final_output = self.meta_learner(meta_input)
        else:
            weights = F.softmax(self.adaptive_weights, dim=0)
            weighted_output = sum(w * m for w, m in zip(weights, model_outputs))
            final_output = self.fusion_proj(weighted_output.view(weighted_output.size(0), -1))
        
        return final_output, {
            'model_outputs': model_outputs,
            'confidences': confidences,
            'adaptive_weights': F.softmax(self.adaptive_weights, dim=0),
            'attention_weights': attn_weights if self.use_attention_fusion else None
        }


class StackingEnsemble(nn.Module):
    def __init__(self,
                 base_models: List[nn.Module],
                 meta_model: Optional[nn.Module] = None,
                 num_classes: int = 2):
        super().__init__()
        
        self.base_models = nn.ModuleList(base_models)
        
        if meta_model is None:
            self.meta_model = nn.Sequential(
                nn.Linear(len(base_models) * num_classes, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
        else:
            self.meta_model = meta_model
    
    def forward(self, inputs: Dict) -> torch.Tensor:
        base_predictions = []
        
        for model in self.base_models:
            if 'pixel_values' in inputs:
                output = model(inputs['pixel_values'])
            elif 'x' in inputs and 'edge_index' in inputs:
                output, _ = model(inputs['x'], inputs['edge_index'], inputs.get('batch'))
            else:
                output = model(inputs.get('data', inputs))
            
            if isinstance(output, tuple):
                output = output[0]
            
            probs = F.softmax(output, dim=1)
            base_predictions.append(probs)
        
        stacked = torch.cat(base_predictions, dim=1)
        final_output = self.meta_model(stacked)
        
        return final_output


class DynamicEnsemble(nn.Module):
    def __init__(self,
                 models: List[nn.Module],
                 num_classes: int = 2,
                 selection_strategy: str = "confidence"):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
        self.selection_strategy = selection_strategy
        
        self.performance_tracker = {i: {'correct': 0, 'total': 0} for i in range(len(models))}
        self.confidence_threshold = 0.7
        
        self.selector = nn.Sequential(
            nn.Linear(num_classes * len(models), 64),
            nn.ReLU(),
            nn.Linear(64, len(models)),
            nn.Softmax(dim=1)
        )
    
    def forward(self, inputs: Dict, update_performance: bool = False) -> Tuple[torch.Tensor, Dict]:
        all_outputs = []
        all_confidences = []
        
        for i, model in enumerate(self.models):
            if 'pixel_values' in inputs:
                output = model(inputs['pixel_values'])
            elif 'x' in inputs and 'edge_index' in inputs:
                output, _ = model(inputs['x'], inputs['edge_index'], inputs.get('batch'))
            else:
                output = model(inputs.get('data', inputs))
            
            if isinstance(output, tuple):
                output = output[0]
            
            probs = F.softmax(output, dim=1)
            confidence = probs.max(dim=1)[0]
            
            all_outputs.append(probs)
            all_confidences.append(confidence)
        
        all_outputs = torch.stack(all_outputs, dim=1)
        all_confidences = torch.stack(all_confidences, dim=1)
        
        if self.selection_strategy == "confidence":
            selected_indices = all_confidences.argmax(dim=1)
            selected_outputs = all_outputs[torch.arange(all_outputs.size(0)), selected_indices]
        elif self.selection_strategy == "weighted":
            stacked = all_outputs.view(all_outputs.size(0), -1)
            selection_weights = self.selector(stacked)
            selected_outputs = (all_outputs * selection_weights.unsqueeze(-1)).sum(dim=1)
        else:
            performance_scores = torch.tensor([
                self.performance_tracker[i]['correct'] / max(1, self.performance_tracker[i]['total'])
                for i in range(len(self.models))
            ]).to(all_outputs.device)
            weights = F.softmax(performance_scores, dim=0)
            selected_outputs = (all_outputs * weights.view(-1, 1, 1)).sum(dim=1)
        
        return selected_outputs, {
            'selected_indices': selected_indices if self.selection_strategy == "confidence" else None,
            'selection_weights': selection_weights if self.selection_strategy == "weighted" else None,
            'all_outputs': all_outputs,
            'all_confidences': all_confidences
        }
    
    def update_performance(self, model_idx: int, is_correct: bool):
        self.performance_tracker[model_idx]['total'] += 1
        if is_correct:
            self.performance_tracker[model_idx]['correct'] += 1


class HierarchicalEnsemble(nn.Module):
    def __init__(self,
                 level1_models: List[nn.Module],
                 level2_models: List[nn.Module],
                 num_classes: int = 2):
        super().__init__()
        
        self.level1_models = nn.ModuleList(level1_models)
        self.level2_models = nn.ModuleList(level2_models)
        
        self.level1_fusion = nn.Linear(len(level1_models) * num_classes, num_classes)
        self.level2_fusion = nn.Linear(len(level2_models) * num_classes, num_classes)
        
        self.final_fusion = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, inputs: Dict) -> torch.Tensor:
        level1_outputs = []
        for model in self.level1_models:
            if 'pixel_values' in inputs:
                output = model(inputs['pixel_values'])
            else:
                output = model(inputs.get('data', inputs))
            
            if isinstance(output, tuple):
                output = output[0]
            
            probs = F.softmax(output, dim=1)
            level1_outputs.append(probs)
        
        level1_stacked = torch.cat(level1_outputs, dim=1)
        level1_fused = self.level1_fusion(level1_stacked)
        
        level2_outputs = []
        for model in self.level2_models:
            if 'x' in inputs and 'edge_index' in inputs:
                output, _ = model(inputs['x'], inputs['edge_index'], inputs.get('batch'))
            else:
                output = model(inputs.get('data', inputs))
            
            if isinstance(output, tuple):
                output = output[0]
            
            probs = F.softmax(output, dim=1)
            level2_outputs.append(probs)
        
        level2_stacked = torch.cat(level2_outputs, dim=1)
        level2_fused = self.level2_fusion(level2_stacked)
        
        combined = torch.cat([level1_fused, level2_fused], dim=1)
        final_output = self.final_fusion(combined)
        
        return final_output


class BayesianEnsemble(nn.Module):
    def __init__(self,
                 models: List[nn.Module],
                 num_classes: int = 2,
                 prior_weight: float = 0.5):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
        self.prior_weight = prior_weight
        
        self.prior = nn.Parameter(torch.ones(num_classes) / num_classes)
        self.model_weights = nn.Parameter(torch.ones(len(models)) / len(models))
        
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
    
    def forward(self, inputs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        model_outputs = []
        uncertainties = []
        
        for model in self.models:
            if 'pixel_values' in inputs:
                output = model(inputs['pixel_values'])
            elif 'x' in inputs and 'edge_index' in inputs:
                output, _ = model(inputs['x'], inputs['edge_index'], inputs.get('batch'))
            else:
                output = model(inputs.get('data', inputs))
            
            if isinstance(output, tuple):
                output = output[0]
            
            probs = F.softmax(output, dim=1)
            model_outputs.append(probs)
            
            uncertainty = self.uncertainty_estimator(output)
            uncertainties.append(uncertainty.squeeze())
        
        model_outputs = torch.stack(model_outputs, dim=1)
        uncertainties = torch.stack(uncertainties, dim=1)
        
        weights = F.softmax(self.model_weights, dim=0)
        inverse_uncertainty = 1.0 / (uncertainties + 1e-8)
        normalized_weights = (weights.unsqueeze(0) * inverse_uncertainty) / (
            (weights.unsqueeze(0) * inverse_uncertainty).sum(dim=1, keepdim=True) + 1e-8
        )
        
        weighted_output = (model_outputs * normalized_weights.unsqueeze(-1)).sum(dim=1)
        
        prior_expanded = self.prior.unsqueeze(0).expand(weighted_output.size(0), -1)
        final_output = self.prior_weight * prior_expanded + (1 - self.prior_weight) * weighted_output
        
        epistemic_uncertainty = uncertainties.mean(dim=1)
        aleatoric_uncertainty = (model_outputs * (1 - model_outputs)).mean(dim=1).mean(dim=1)
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return final_output, total_uncertainty
