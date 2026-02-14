import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch, Data
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class GCNFraudDetector(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_classes: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        
        if batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        else:
            x_mean = x.mean(dim=0, keepdim=True)
            x_max = x.max(dim=0, keepdim=True)[0]
            x = torch.cat([x_mean, x_max], dim=1)
        
        x = self.classifier(x)
        return x


class GraphSAGEFraudDetector(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_classes: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.gnn = GraphSAGE(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=hidden_dim,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.gnn(x, edge_index)
        
        if batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        else:
            x_mean = x.mean(dim=0, keepdim=True)
            x_max = x.max(dim=0, keepdim=True)[0]
            x = torch.cat([x_mean, x_max], dim=1)
        
        x = self.classifier(x)
        return x


class GATFraudDetector(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 num_classes: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, concat=False))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False) -> Tuple[torch.Tensor, Optional[List]]:
        attention_weights = []
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            if return_attention_weights and i < len(self.convs) - 1:
                x, attn = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append(attn)
            else:
                x = conv(x, edge_index)
            
            x = bn(x)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        
        if batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        else:
            x_mean = x.mean(dim=0, keepdim=True)
            x_max = x.max(dim=0, keepdim=True)[0]
            x = torch.cat([x_mean, x_max], dim=1)
        
        x = self.classifier(x)
        
        if return_attention_weights:
            return x, attention_weights
        return x, None


class GINFraudDetector(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_classes: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn1))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            nn_mid = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_mid))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        nn_final = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn_final))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        
        if batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        else:
            x_mean = x.mean(dim=0, keepdim=True)
            x_max = x.max(dim=0, keepdim=True)[0]
            x = torch.cat([x_mean, x_max], dim=1)
        
        x = self.classifier(x)
        return x


class TemporalGNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_classes: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        from torch_geometric.nn import GCNConv
        from torch_geometric.nn import GatedGraphConv
        
        self.gated_conv = GatedGraphConv(hidden_dim, num_layers)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.temporal_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                temporal_features: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.gated_conv(x, edge_index)
        
        if temporal_features is not None:
            x = x.unsqueeze(1)
            lstm_out, _ = self.temporal_lstm(x)
            x = lstm_out.squeeze(1)
        
        if batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        else:
            x_mean = x.mean(dim=0, keepdim=True)
            x_max = x.max(dim=0, keepdim=True)[0]
            x = torch.cat([x_mean, x_max], dim=1)
        
        x = self.classifier(x)
        return x
