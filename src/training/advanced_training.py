import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts,
    ReduceLROnPlateau, OneCycleLR, LambdaLR
)
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import copy

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class AdvancedTrainer:
    def __init__(self,
                 model: nn.Module,
                 config: Dict,
                 device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        
        self.scaler = GradScaler() if config.get('use_mixed_precision', False) else None
        self.use_mixed_precision = config.get('use_mixed_precision', False)
        
        self.best_model_state = None
        self.best_score = -float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def _create_optimizer(self):
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adamw')
        lr = optimizer_config.get('learning_rate', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 0.01)
        
        if optimizer_type == 'adamw':
            return AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            return SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            return RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _create_scheduler(self):
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', 10)
            return CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_type == 'cosine_restarts':
            T_0 = scheduler_config.get('T_0', 10)
            T_mult = scheduler_config.get('T_mult', 2)
            return CosineAnnealingWarmRestarts(self.optimizer, T_0=T_0, T_mult=T_mult)
        elif scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3)
        elif scheduler_type == 'onecycle':
            max_lr = scheduler_config.get('max_lr', 1e-3)
            steps_per_epoch = scheduler_config.get('steps_per_epoch', 100)
            epochs = scheduler_config.get('epochs', 10)
            return OneCycleLR(self.optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
        else:
            return None
    
    def _create_criterion(self):
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'cross_entropy')
        
        if loss_type == 'focal':
            return FocalLoss(alpha=loss_config.get('alpha', 1.0), gamma=loss_config.get('gamma', 2.0))
        elif loss_type == 'label_smoothing':
            num_classes = loss_config.get('num_classes', 2)
            smoothing = loss_config.get('smoothing', 0.1)
            return LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
        else:
            return nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            inputs = self._prepare_inputs(batch)
            targets = batch['target'].to(self.device)
            
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(**inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                
                if self.config.get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('gradient_clip', 1.0)
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(**inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                
                if self.config.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('gradient_clip', 1.0)
                    )
                
                self.optimizer.step()
            
            if self.scheduler and isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, val_loader) -> Dict:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs = self._prepare_inputs(batch)
                targets = batch['target'].to(self.device)
                
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(**inputs)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(**inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
                
                all_predictions.append(pred.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        precision = precision_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _prepare_inputs(self, batch: Dict) -> Dict:
        inputs = {}
        
        if 'pixel_values' in batch:
            inputs['pixel_values'] = batch['pixel_values'].to(self.device)
        
        if 'input_ids' in batch:
            inputs['input_ids'] = batch['input_ids'].to(self.device)
            if 'attention_mask' in batch:
                inputs['attention_mask'] = batch['attention_mask'].to(self.device)
        
        if 'x' in batch and 'edge_index' in batch:
            inputs['x'] = batch['x'].to(self.device)
            inputs['edge_index'] = batch['edge_index'].to(self.device)
            if 'batch' in batch:
                inputs['batch'] = batch['batch'].to(self.device)
        
        if 'data' in batch:
            inputs['data'] = batch['data'].to(self.device)
        
        return inputs
    
    def train(self, train_loader, val_loader, num_epochs: int):
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('f1_score', val_metrics['accuracy']))
                elif not isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
            
            score = val_metrics.get('f1_score', val_metrics['accuracy'])
            if score > self.best_score:
                self.best_score = score
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                logger.info(f"New best model! Score: {score:.4f}")
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}% - "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, "
                f"F1: {val_metrics.get('f1_score', 0):.4f}"
            )
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return self.training_history
