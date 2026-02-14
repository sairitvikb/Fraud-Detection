import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import random
from collections import namedtuple
import logging

logger = logging.getLogger(__name__)

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


class NASCell(nn.Module):
    def __init__(self, C_prev_prev: int, C_prev: int, C: int, reduction: bool, reduction_prev: bool):
        super().__init__()
        self.reduction = reduction
        
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        
        if reduction:
            op_names, indices = zip(*[('max_pool_3x3', 0), ('avg_pool_3x3', 1),
                                      ('skip_connect', 2), ('sep_conv_3x3', 3),
                                      ('sep_conv_5x5', 4), ('dil_conv_3x3', 5),
                                      ('dil_conv_5x5', 6)])
        else:
            op_names, indices = zip(*[('max_pool_3x3', 0), ('avg_pool_3x3', 1),
                                      ('skip_connect', 2), ('sep_conv_3x3', 3),
                                      ('sep_conv_5x5', 4), ('dil_conv_3x3', 5),
                                      ('dil_conv_5x5', 6)])
        
        self._compile(C, op_names, reduction)
    
    def _compile(self, C: int, op_names: Tuple, reduction: bool):
        assert len(op_names) == len(self._multiplier) * len(self._steps)
        self._ops = nn.ModuleList()
        for name in op_names:
            stride = 2 if reduction and name.startswith('down') else 1
            op = OPS[name](C, stride, True)
            self._ops.append(op)
    
    def forward(self, s0: torch.Tensor, s1: torch.Tensor, drop_prob: float) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states.append(s)
        return torch.cat([states[i] for i in self._concat], dim=1)


class DARTS(nn.Module):
    def __init__(self, C: int, num_classes: int, layers: int, genotype: Genotype):
        super().__init__()
        self._layers = layers
        self._genotype = genotype
        
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = NASCell(C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


class ProgressiveNAS:
    def __init__(self, search_space: Dict, population_size: int = 50):
        self.search_space = search_space
        self.population_size = population_size
        self.population = []
        self.fitness_history = []
    
    def initialize_population(self):
        for _ in range(self.population_size):
            architecture = self._random_architecture()
            self.population.append(architecture)
    
    def _random_architecture(self) -> Dict:
        architecture = {}
        for key, options in self.search_space.items():
            architecture[key] = random.choice(options)
        return architecture
    
    def evolve(self, fitness_fn, generations: int = 20, mutation_rate: float = 0.1):
        for generation in range(generations):
            fitness_scores = [fitness_fn(arch) for arch in self.population]
            self.fitness_history.append(max(fitness_scores))
            
            sorted_pop = sorted(zip(self.population, fitness_scores), key=lambda x: x[1], reverse=True)
            
            elite_size = self.population_size // 4
            elite = [arch for arch, _ in sorted_pop[:elite_size]]
            
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(elite, 2)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child, mutation_rate)
                new_population.append(child)
            
            self.population = new_population
        
        return max(self.population, key=fitness_fn)
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        child = {}
        for key in parent1:
            child[key] = random.choice([parent1[key], parent2[key]])
        return child
    
    def _mutate(self, architecture: Dict, mutation_rate: float) -> Dict:
        mutated = architecture.copy()
        for key in mutated:
            if random.random() < mutation_rate:
                mutated[key] = random.choice(self.search_space[key])
        return mutated


class DifferentiableNAS:
    def __init__(self, num_ops: int = 8, num_nodes: int = 4):
        self.num_ops = num_ops
        self.num_nodes = num_nodes
        self.alpha_normal = nn.Parameter(torch.randn(num_nodes, num_ops))
        self.alpha_reduce = nn.Parameter(torch.randn(num_nodes, num_ops))
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        if training:
            weights_normal = F.softmax(self.alpha_normal, dim=-1)
            weights_reduce = F.softmax(self.alpha_reduce, dim=-1)
        else:
            weights_normal = F.one_hot(self.alpha_normal.argmax(dim=-1), self.num_ops).float()
            weights_reduce = F.one_hot(self.alpha_reduce.argmax(dim=-1), self.num_ops).float()
        
        return x
    
    def genotype(self) -> Genotype:
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self.num_nodes):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene
        
        gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alpha_reduce, dim=-1).data.cpu().numpy())
        
        concat = range(2, 2 + self.num_nodes)
        return Genotype(normal=gene_normal, normal_concat=concat,
                       reduce=gene_reduce, reduce_concat=concat)


PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]


def drop_path(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).to(x.device)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class ReLUConvBN(nn.Module):
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in: int, C_out: int):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
}


class Zero(nn.Module):
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SepConv(nn.Module):
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int, affine: bool = True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in: int, C_out: int, kernel_size: int, stride: int, padding: int, dilation: int, affine: bool = True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)
