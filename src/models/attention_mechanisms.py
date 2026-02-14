import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        
        return output, attention_weights


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x, mask)
        output = self.norm(x + self.dropout(attn_output))
        return output


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attention(query, key, value, mask)
        output = self.norm(query + self.dropout(attn_output))
        return output


class SparseAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, sparsity: float = 0.5):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.sparsity = sparsity
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)
        
        attn_output, attention_weights = self.attention(query, key, value)
        
        k = max(1, int(seq_len * self.sparsity))
        top_k_weights, top_k_indices = torch.topk(attention_weights, k, dim=-1)
        
        sparse_weights = torch.zeros_like(attention_weights)
        sparse_weights.scatter_(-1, top_k_indices, top_k_weights)
        sparse_weights = F.softmax(sparse_weights, dim=-1)
        
        return attn_output


class TemporalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.temporal_encoding = nn.Parameter(torch.randn(1000, d_model))
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        
        temporal_emb = self.temporal_encoding[positions]
        x = x + temporal_emb.unsqueeze(0)
        
        attn_output, _ = self.attention(x, x, x)
        return attn_output
