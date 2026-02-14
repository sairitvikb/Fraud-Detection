import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

class MAML(nn.Module):
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, num_inner_steps: int = 1):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
    
    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor,
                query_x: torch.Tensor, query_y: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        fast_weights = dict(self.model.named_parameters())
        
        for _ in range(self.num_inner_steps):
            support_logits = self._forward_with_weights(support_x, fast_weights)
            loss = F.cross_entropy(support_logits, support_y)
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            fast_weights = {name: param - self.inner_lr * grad
                          for (name, param), grad in zip(fast_weights.items(), grads)}
        
        query_logits = self._forward_with_weights(query_x, fast_weights)
        query_loss = F.cross_entropy(query_logits, query_y)
        
        return query_logits, {'loss': query_loss}
    
    def _forward_with_weights(self, x: torch.Tensor, weights: Dict) -> torch.Tensor:
        return self.model(x)


class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_dim: int = 64):
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
    
    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor,
                query_x: torch.Tensor) -> torch.Tensor:
        support_embeddings = self.encoder(support_x)
        query_embeddings = self.encoder(query_x)
        
        unique_labels = torch.unique(support_y)
        prototypes = []
        
        for label in unique_labels:
            label_mask = (support_y == label)
            label_embeddings = support_embeddings[label_mask]
            prototype = label_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        
        distances = torch.cdist(query_embeddings, prototypes)
        logits = -distances
        
        return logits


class RelationNetwork(nn.Module):
    def __init__(self, encoder: nn.Module, relation_network: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.relation_network = relation_network
    
    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor,
                query_x: torch.Tensor) -> torch.Tensor:
        support_embeddings = self.encoder(support_x)
        query_embeddings = self.encoder(query_x)
        
        unique_labels = torch.unique(support_y)
        prototypes = []
        
        for label in unique_labels:
            label_mask = (support_y == label)
            label_embeddings = support_embeddings[label_mask]
            prototype = label_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        
        batch_size = query_embeddings.size(0)
        num_prototypes = prototypes.size(0)
        
        query_expanded = query_embeddings.unsqueeze(1).expand(-1, num_prototypes, -1)
        prototypes_expanded = prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        
        combined = torch.cat([query_expanded, prototypes_expanded], dim=2)
        relations = self.relation_network(combined).squeeze(-1)
        
        return relations


class MatchingNetwork(nn.Module):
    def __init__(self, encoder: nn.Module, attention: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.attention = attention
    
    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor,
                query_x: torch.Tensor) -> torch.Tensor:
        support_embeddings = self.encoder(support_x)
        query_embeddings = self.encoder(query_x)
        
        attention_weights = self.attention(query_embeddings, support_embeddings)
        
        attended_support = torch.matmul(attention_weights, support_embeddings)
        
        similarities = F.cosine_similarity(
            query_embeddings.unsqueeze(1),
            attended_support.unsqueeze(0),
            dim=2
        )
        
        return similarities
