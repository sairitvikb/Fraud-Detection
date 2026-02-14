import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig, ViTForImageClassification
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VisionTransformerFraudDetector(nn.Module):
    def __init__(self, 
                 model_name: str = "google/vit-base-patch16-224",
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 pretrained: bool = True):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        config = ViTConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        config.image_size = image_size
        config.patch_size = patch_size
        
        if pretrained:
            self.vit = ViTForImageClassification.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True
            )
        else:
            self.vit = ViTForImageClassification(config)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        
        if hasattr(self.vit, 'classifier'):
            self.vit.classifier = nn.Identity()
        
    def forward(self, pixel_values: torch.Tensor, 
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = self.vit.vit(pixel_values, output_attentions=return_attention)
        
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if return_attention:
            return logits, outputs.attentions
        return logits, None
    
    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.vit.vit(pixel_values)
            return outputs.last_hidden_state[:, 0]


class MultiScaleVisionTransformer(nn.Module):
    def __init__(self,
                 model_names: list = None,
                 num_classes: int = 2,
                 fusion_method: str = "attention"):
        super().__init__()
        
        if model_names is None:
            model_names = [
                "google/vit-base-patch16-224",
                "google/vit-large-patch16-224"
            ]
        
        self.models = nn.ModuleList([
            VisionTransformerFraudDetector(model_name=name, num_classes=num_classes)
            for name in model_names
        ])
        
        self.fusion_method = fusion_method
        
        if fusion_method == "attention":
            self.attention_weights = nn.MultiheadAttention(
                embed_dim=768,
                num_heads=8,
                batch_first=True
            )
            self.fusion_layer = nn.Linear(768 * len(model_names), 768)
        elif fusion_method == "weighted":
            self.fusion_weights = nn.Parameter(torch.ones(len(model_names)) / len(model_names))
        else:
            self.fusion_layer = nn.Linear(768 * len(model_names), 768)
        
        self.classifier = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = []
        
        for model in self.models:
            feat = model.extract_features(pixel_values)
            features.append(feat)
        
        if self.fusion_method == "attention":
            stacked_features = torch.stack(features, dim=1)
            attn_output, _ = self.attention_weights(
                stacked_features, stacked_features, stacked_features
            )
            fused = attn_output.mean(dim=1)
            fused = self.fusion_layer(fused.flatten(1))
        elif self.fusion_method == "weighted":
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = sum(w * f for w, f in zip(weights, features))
        else:
            concatenated = torch.cat(features, dim=1)
            fused = self.fusion_layer(concatenated)
        
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        
        return logits


class VisionTransformerWithAuxiliary(nn.Module):
    def __init__(self,
                 model_name: str = "google/vit-base-patch16-224",
                 num_classes: int = 2,
                 auxiliary_features_dim: int = 10):
        super().__init__()
        
        self.vit = VisionTransformerFraudDetector(
            model_name=model_name,
            num_classes=num_classes
        )
        
        self.auxiliary_encoder = nn.Sequential(
            nn.Linear(auxiliary_features_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(768 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, pixel_values: torch.Tensor, 
                auxiliary_features: torch.Tensor) -> torch.Tensor:
        vit_features = self.vit.extract_features(pixel_values)
        aux_features = self.auxiliary_encoder(auxiliary_features)
        
        combined = torch.cat([vit_features, aux_features], dim=1)
        logits = self.fusion(combined)
        
        return logits
