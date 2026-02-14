import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertModel, RobertaModel, DebertaModel,
    GPT2Model, T5EncoderModel,
    ViTModel, SwinModel, ConvNextModel
)
from typing import Optional, Tuple, Dict, List
import logging
import math

logger = logging.getLogger(__name__)


class TransformerEncoderFraudDetector(nn.Module):
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits


class MultiModalTransformer(nn.Module):
    def __init__(self,
                 text_model_name: str = "bert-base-uncased",
                 image_model_name: str = "google/vit-base-patch16-224",
                 num_classes: int = 2,
                 fusion_method: str = "cross_attention"):
        super().__init__()
        
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.image_encoder = AutoModel.from_pretrained(image_model_name)
        
        text_dim = self.text_encoder.config.hidden_size
        image_dim = self.image_encoder.config.hidden_size
        
        self.fusion_method = fusion_method
        
        if fusion_method == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=text_dim,
                num_heads=8,
                batch_first=True
            )
            self.fusion = nn.Linear(text_dim + image_dim, text_dim)
        elif fusion_method == "bilinear":
            self.fusion = nn.Bilinear(text_dim, image_dim, text_dim)
        else:
            self.fusion = nn.Linear(text_dim + image_dim, text_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self,
               input_ids: torch.Tensor,
               pixel_values: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state
        
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_features = image_outputs.last_hidden_state[:, 0]
        
        if self.fusion_method == "cross_attention":
            image_features_expanded = image_features.unsqueeze(1).expand(-1, text_features.size(1), -1)
            fused, _ = self.cross_attention(text_features, image_features_expanded, image_features_expanded)
            fused = fused.mean(dim=1)
            combined = torch.cat([fused, image_features], dim=1)
            fused_features = self.fusion(combined)
        elif self.fusion_method == "bilinear":
            text_pooled = text_features.mean(dim=1)
            fused_features = self.fusion(text_pooled, image_features)
        else:
            text_pooled = text_features.mean(dim=1)
            combined = torch.cat([text_pooled, image_features], dim=1)
            fused_features = self.fusion(combined)
        
        logits = self.classifier(fused_features)
        return logits


class TemporalTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 num_classes: int = 2,
                 max_seq_length: int = 512):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_length, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        
        return logits


class HierarchicalTransformer(nn.Module):
    def __init__(self,
                 base_model_name: str = "bert-base-uncased",
                 num_classes: int = 2,
                 num_hierarchical_levels: int = 3):
        super().__init__()
        
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        
        self.hierarchical_encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=8,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=2
            ) for _ in range(num_hierarchical_levels)
        ])
        
        self.level_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_hierarchical_levels)
        ])
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * num_hierarchical_levels, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        base_features = base_outputs.last_hidden_state
        
        hierarchical_features = []
        current_features = base_features
        
        for encoder, projection in zip(self.hierarchical_encoders, self.level_projections):
            encoded = encoder(current_features)
            projected = projection(encoded.mean(dim=1))
            hierarchical_features.append(projected)
            current_features = encoded
        
        fused = torch.cat(hierarchical_features, dim=1)
        logits = self.fusion(fused)
        
        return logits


class GraphTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 num_classes: int = 2):
        super().__init__()
        
        from torch_geometric.nn import TransformerConv
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.transformer_convs = nn.ModuleList([
            TransformerConv(
                hidden_dim,
                hidden_dim,
                heads=num_heads,
                dropout=0.1,
                concat=True
            ) for _ in range(num_layers - 1)
        ])
        
        self.final_conv = TransformerConv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=1,
            dropout=0.1,
            concat=False
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        from torch_geometric.nn import global_mean_pool
        
        x = self.input_proj(x)
        
        for conv in self.transformer_convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.final_conv(x, edge_index)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        logits = self.classifier(x)
        return logits


class AdversarialRobustTransformer(nn.Module):
    def __init__(self,
                 base_model_name: str = "bert-base-uncased",
                 num_classes: int = 2,
                 adversarial_training: bool = True):
        super().__init__()
        
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.adversarial_training = adversarial_training
        if adversarial_training:
            self.adversarial_layer = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                adversarial: bool = False) -> torch.Tensor:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        features = outputs.last_hidden_state[:, 0]
        
        if adversarial and self.adversarial_training:
            features = features + 0.01 * self.adversarial_layer(features)
        
        logits = self.classifier(features)
        return logits
    
    def generate_adversarial_examples(self, input_ids: torch.Tensor,
                                     attention_mask: Optional[torch.Tensor] = None,
                                     epsilon: float = 0.1) -> torch.Tensor:
        input_ids = input_ids.clone()
        input_ids.requires_grad = True
        
        outputs = self.forward(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, outputs.argmax(dim=1))
        loss.backward()
        
        perturbation = epsilon * input_ids.grad.sign()
        adversarial_ids = input_ids + perturbation.long()
        
        return adversarial_ids
