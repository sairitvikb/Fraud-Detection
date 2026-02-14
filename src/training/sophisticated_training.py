import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts,
    ReduceLROnPlateau, OneCycleLR, LambdaLR, CyclicLR
)
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import copy
from collections import defaultdict
import math
from ..models.advanced_architectures import LabelSmoothingCrossEntropy, FocalLoss, PolyLoss

logger = logging.getLogger(__name__)


class LookaheadOptimizer:
    def __init__(self, optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        self.slow_weights = {param: param.data.clone() for param in self.optimizer.param_groups[0]['params']}
    
    def step(self):
        self.optimizer.step()
        self.step_count += 1
        
        if self.step_count % self.k == 0:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p in self.slow_weights:
                        p.data.mul_(self.alpha).add_(self.slow_weights[p], alpha=1 - self.alpha)
                        self.slow_weights[p].copy_(p.data)
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


class RAdamOptimizer:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.state = defaultdict(dict)
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        beta1, beta2 = self.betas
        
        for p in self.params:
            if p.grad is None:
                continue
            
            grad = p.grad.data
            if self.weight_decay != 0:
                grad = grad.add(p.data, alpha=self.weight_decay)
            
            state = self.state[p]
            
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            state['step'] += 1
            
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            beta2_t = beta2 ** state['step']
            N_sma_max = 2 / (1 - beta2) - 1
            N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
            
            if N_sma >= 5:
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * state['step'] * beta2_t / (1 - beta2_t)
                rho_inf = max(rho_inf, 4)
                
                if rho_t > 4:
                    rho_t = max(rho_t, 4)
                    r_t = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                    
                    p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(self.eps), value=-self.lr * r_t * (1 - beta1 ** state['step']))
                else:
                    p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(self.eps), value=-self.lr * (1 - beta1 ** state['step']))
            else:
                p.data.add_(exp_avg, alpha=-self.lr * (1 - beta1 ** state['step']))
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


class GradientAccumulator:
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_step(self) -> bool:
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0
    
    def reset(self):
        self.current_step = 0


class AdvancedTrainer:
    def __init__(self, model: nn.Module, config: Dict, device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        
        self.scaler = GradScaler() if config.get('use_mixed_precision', False) else None
        self.use_mixed_precision = config.get('use_mixed_precision', False)
        
        self.gradient_accumulator = GradientAccumulator(
            config.get('gradient_accumulation_steps', 1)
        )
        
        self.ema_model = copy.deepcopy(model)
        self.ema_decay = config.get('ema_decay', 0.999)
        self.use_ema = config.get('use_ema', False)
        
        self.best_model_state = None
        self.best_score = -float('inf')
        self.training_history = defaultdict(list)
        
        self.label_smoothing = config.get('label_smoothing', 0.0)
        self.cutmix_prob = config.get('cutmix_prob', 0.0)
        self.mixup_prob = config.get('mixup_prob', 0.0)
        self.mixup_alpha = config.get('mixup_alpha', 1.0)
    
    def _create_optimizer(self):
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adamw')
        lr = optimizer_config.get('learning_rate', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 0.01)
        
        if optimizer_type == 'adamw':
            optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay,
                             betas=(0.9, 0.999), eps=1e-8)
        elif optimizer_type == 'radam':
            optimizer = RAdamOptimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            optimizer = SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
                          nesterov=True)
        else:
            optimizer = RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        if optimizer_config.get('use_lookahead', False):
            optimizer = LookaheadOptimizer(optimizer, k=optimizer_config.get('lookahead_k', 5))
        
        return optimizer
    
    def _create_scheduler(self):
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', 10)
            eta_min = scheduler_config.get('eta_min', 0)
            return CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type == 'cosine_restarts':
            T_0 = scheduler_config.get('T_0', 10)
            T_mult = scheduler_config.get('T_mult', 2)
            return CosineAnnealingWarmRestarts(self.optimizer, T_0=T_0, T_mult=T_mult)
        elif scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        elif scheduler_type == 'onecycle':
            max_lr = scheduler_config.get('max_lr', 1e-3)
            steps_per_epoch = scheduler_config.get('steps_per_epoch', 100)
            epochs = scheduler_config.get('epochs', 10)
            return OneCycleLR(self.optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch,
                            epochs=epochs, pct_start=0.3)
        elif scheduler_type == 'cyclic':
            base_lr = scheduler_config.get('base_lr', 1e-5)
            max_lr = scheduler_config.get('max_lr', 1e-3)
            return CyclicLR(self.optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=2000)
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
            return LabelSmoothingCrossEntropy(smoothing=smoothing)
        elif loss_type == 'poly':
            return PolyLoss(epsilon=loss_config.get('epsilon', 1.0))
        else:
            return nn.CrossEntropyLoss()
    
    def _apply_mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        if np.random.rand() > self.mixup_prob:
            return x, y, 1.0, y
        
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, lam, y_b
    
    def _apply_cutmix(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        if np.random.rand() > self.cutmix_prob:
            return x, y, 1.0, y
        
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        W, H = x.size(3), x.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        
        return x, y_a, lam, y_b
    
    def _update_ema(self):
        if not self.use_ema:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def train_epoch(self, train_loader, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        self.gradient_accumulator.reset()
        
        for batch_idx, batch in enumerate(pbar):
            inputs = self._prepare_inputs(batch)
            targets = batch['target'].to(self.device)
            
            if self.mixup_prob > 0 and np.random.rand() < 0.5:
                inputs['data'], targets, lam, targets_b = self._apply_mixup(
                    inputs.get('data', inputs.get('pixel_values', inputs.get('x'))),
                    targets
                )
            elif self.cutmix_prob > 0:
                inputs['data'], targets, lam, targets_b = self._apply_cutmix(
                    inputs.get('data', inputs.get('pixel_values', inputs.get('x'))),
                    targets
                )
            else:
                lam = 1.0
                targets_b = None
            
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(**inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    if targets_b is not None:
                        loss = lam * self.criterion(outputs, targets) + (1 - lam) * self.criterion(outputs, targets_b)
                    else:
                        loss = self.criterion(outputs, targets)
                    
                    loss = loss / self.gradient_accumulator.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if self.gradient_accumulator.should_step():
                    if self.config.get('gradient_clip', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.get('gradient_clip', 1.0)
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self._update_ema()
            else:
                outputs = self.model(**inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if targets_b is not None:
                    loss = lam * self.criterion(outputs, targets) + (1 - lam) * self.criterion(outputs, targets_b)
                else:
                    loss = self.criterion(outputs, targets)
                
                loss = loss / self.gradient_accumulator.accumulation_steps
                loss.backward()
                
                if self.gradient_accumulator.should_step():
                    if self.config.get('gradient_clip', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.get('gradient_clip', 1.0)
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self._update_ema()
            
            if self.scheduler and isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            total_loss += loss.item() * self.gradient_accumulator.accumulation_steps
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
            
            pbar.set_postfix({
                'loss': loss.item() * self.gradient_accumulator.accumulation_steps,
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, val_loader, use_ema: bool = False) -> Dict:
        model = self.ema_model if use_ema and self.use_ema else self.model
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs = self._prepare_inputs(batch)
                targets = batch['target'].to(self.device)
                
                if self.use_mixed_precision:
                    with autocast():
                        outputs = model(**inputs)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = model(**inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                probs = F.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
                
                all_predictions.append(pred.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)
        
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
        
        precision = precision_score(all_targets, all_predictions, zero_division=0, average='weighted')
        recall = recall_score(all_targets, all_predictions, zero_division=0, average='weighted')
        f1 = f1_score(all_targets, all_predictions, zero_division=0, average='weighted')
        
        try:
            roc_auc = roc_auc_score(all_targets, all_probs[:, 1])
            pr_auc = average_precision_score(all_targets, all_probs[:, 1])
        except:
            roc_auc = 0.0
            pr_auc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
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
            val_metrics = self.validate(val_loader, use_ema=False)
            
            if self.use_ema:
                ema_val_metrics = self.validate(val_loader, use_ema=True)
                logger.info(f"EMA Val Metrics: {ema_val_metrics}")
            
            for key, value in train_metrics.items():
                self.training_history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.training_history[f'val_{key}'].append(value)
            
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
                f"F1: {val_metrics.get('f1_score', 0):.4f}, ROC-AUC: {val_metrics.get('roc_auc', 0):.4f}"
            )
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return self.training_history
