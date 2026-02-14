import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging
from .vision_transformer import VisionTransformerFraudDetector
from .graph_neural_network import GATFraudDetector, GCNFraudDetector, GraphSAGEFraudDetector

logger = logging.getLogger(__name__)


class MultimodalEnsemble(nn.Module):
    def __init__(self,
                 vit_config: Dict,
                 gnn_config: Dict,
                 ensemble_config: Dict,
                 input_dim: int = 768):
        super().__init__()
        
        self.vit_weight = ensemble_config.get('vit_weight', 0.4)
        self.gnn_weight = ensemble_config.get('gnn_weight', 0.6)
        self.use_adaptive_weights = ensemble_config.get('adaptive_weights', False)
        
        self.vit_model = VisionTransformerFraudDetector(**vit_config)
        
        gnn_arch = gnn_config.get('architecture', 'gat').lower()
        if gnn_arch == 'gat':
            self.gnn_model = GATFraudDetector(input_dim=input_dim, **gnn_config)
        elif gnn_arch == 'gcn':
            self.gnn_model = GCNFraudDetector(input_dim=input_dim, **gnn_config)
        elif gnn_arch == 'graphsage':
            self.gnn_model = GraphSAGEFraudDetector(input_dim=input_dim, **gnn_config)
        else:
            self.gnn_model = GATFraudDetector(input_dim=input_dim, **gnn_config)
        
        if self.use_adaptive_weights:
            self.weight_network = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 2),
                nn.Softmax(dim=1)
            )
        
        self.final_classifier = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def forward(self,
                pixel_values: Optional[torch.Tensor] = None,
                x: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        
        vit_logits = None
        gnn_logits = None
        
        if pixel_values is not None:
            vit_logits, _ = self.vit_model(pixel_values)
            vit_probs = F.softmax(vit_logits, dim=1)
        
        if x is not None and edge_index is not None:
            gnn_logits, _ = self.gnn_model(x, edge_index, batch)
            gnn_probs = F.softmax(gnn_logits, dim=1)
        
        if vit_probs is not None and gnn_probs is not None:
            if self.use_adaptive_weights:
                confidence_scores = torch.stack([
                    vit_probs.max(dim=1)[0],
                    gnn_probs.max(dim=1)[0]
                ], dim=1)
                weights = self.weight_network(confidence_scores)
                combined_probs = weights[:, 0:1] * vit_probs + weights[:, 1:2] * gnn_probs
            else:
                combined_probs = self.vit_weight * vit_probs + self.gnn_weight * gnn_probs
            
            final_logits = self.final_classifier(combined_probs)
        elif vit_probs is not None:
            final_logits = vit_logits
        elif gnn_probs is not None:
            final_logits = gnn_logits
        else:
            raise ValueError("At least one model input must be provided")
        
        return final_logits, {
            'vit_probs': vit_probs if vit_probs is not None else None,
            'gnn_probs': gnn_probs if gnn_probs is not None else None,
            'combined_probs': combined_probs if vit_probs is not None and gnn_probs is not None else None
        }
