import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from transformers import AutoModel, AutoTokenizer
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
from collections import deque

logger = logging.getLogger(__name__)


class AgentType(Enum):
    DETECTOR = "detector"
    VALIDATOR = "validator"
    EXPLAINER = "explainer"
    OPTIMIZER = "optimizer"
    COORDINATOR = "coordinator"


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    message_type: str
    payload: Dict
    timestamp: float
    priority: int = 0


class BaseAgent(nn.Module):
    def __init__(self, agent_id: str, agent_type: AgentType, config: Dict):
        super().__init__()
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.message_queue = deque(maxlen=1000)
        self.state = {}
        self.performance_metrics = {}
    
    def receive_message(self, message: AgentMessage):
        self.message_queue.append(message)
    
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        raise NotImplementedError
    
    def update_state(self, new_state: Dict):
        self.state.update(new_state)
    
    def get_performance_metrics(self) -> Dict:
        return self.performance_metrics


class FraudDetectorAgent(BaseAgent):
    def __init__(self, agent_id: str, model: nn.Module, config: Dict):
        super().__init__(agent_id, AgentType.DETECTOR, config)
        self.model = model
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.detection_history = deque(maxlen=10000)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            fraud_prob = probs[:, 1]
            is_fraud = (fraud_prob > self.confidence_threshold).long()
        
        result = {
            'is_fraud': is_fraud,
            'confidence': fraud_prob,
            'logits': logits,
            'probabilities': probs
        }
        
        self.detection_history.append(result)
        return is_fraud, result
    
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == "detect":
            x = message.payload['data']
            is_fraud, result = self.forward(x)
            
            response = AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type="detection_result",
                payload=result,
                timestamp=message.timestamp,
                priority=1 if is_fraud.item() else 0
            )
            return response
        return None


class ValidationAgent(BaseAgent):
    def __init__(self, agent_id: str, validator_model: nn.Module, config: Dict):
        super().__init__(agent_id, AgentType.VALIDATOR, config)
        self.validator = validator_model
        self.validation_rules = config.get('validation_rules', [])
        self.consensus_threshold = config.get('consensus_threshold', 0.7)
    
    def validate_prediction(self, prediction: Dict, context: Dict) -> Dict:
        validation_score = 0.0
        violations = []
        
        for rule in self.validation_rules:
            if rule['type'] == 'confidence':
                if prediction['confidence'] < rule['threshold']:
                    violations.append(f"Low confidence: {prediction['confidence']:.3f}")
                    validation_score -= 0.1
            elif rule['type'] == 'consistency':
                if 'historical' in context:
                    hist_avg = np.mean(context['historical'])
                    if abs(prediction['confidence'] - hist_avg) > rule['threshold']:
                        violations.append("Inconsistent with history")
                        validation_score -= 0.15
        
        with torch.no_grad():
            validator_input = torch.cat([
                prediction['logits'],
                torch.tensor([prediction['confidence']]).unsqueeze(0)
            ], dim=1)
            validator_output = self.validator(validator_input)
            validator_score = F.sigmoid(validator_output).item()
        
        validation_score += validator_score
        is_valid = validation_score >= self.consensus_threshold
        
        return {
            'is_valid': is_valid,
            'validation_score': validation_score,
            'validator_score': validator_score,
            'violations': violations
        }
    
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == "validate":
            prediction = message.payload['prediction']
            context = message.payload.get('context', {})
            
            validation_result = self.validate_prediction(prediction, context)
            
            response = AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type="validation_result",
                payload=validation_result,
                timestamp=message.timestamp
            )
            return response
        return None


class ExplanationAgent(BaseAgent):
    def __init__(self, agent_id: str, explainer_model, config: Dict):
        super().__init__(agent_id, AgentType.EXPLAINER, config)
        self.explainer = explainer_model
        self.explanation_method = config.get('method', 'shap')
        self.feature_names = config.get('feature_names', [])
    
    def generate_explanation(self, model, data: torch.Tensor, prediction: Dict) -> Dict:
        if self.explanation_method == 'shap':
            explanation = self.explainer.explain(data)
        elif self.explanation_method == 'gradient':
            explanation = self._gradient_explanation(model, data)
        elif self.explanation_method == 'attention':
            explanation = self._attention_explanation(model, data)
        else:
            explanation = self._integrated_gradients(model, data)
        
        top_features = self._get_top_features(explanation, k=10)
        
        return {
            'explanation': explanation,
            'top_features': top_features,
            'feature_importance': explanation.get('shap_values', {}),
            'confidence': prediction.get('confidence', 0.0)
        }
    
    def _gradient_explanation(self, model, data: torch.Tensor) -> Dict:
        data.requires_grad = True
        output = model(data)
        output[:, 1].backward()
        gradients = data.grad.abs().mean(dim=0)
        return {'gradients': gradients.cpu().numpy()}
    
    def _attention_explanation(self, model, data: torch.Tensor) -> Dict:
        if hasattr(model, 'get_attention_weights'):
            attention = model.get_attention_weights(data)
            return {'attention': attention}
        return {}
    
    def _integrated_gradients(self, model, data: torch.Tensor, steps: int = 50) -> Dict:
        baseline = torch.zeros_like(data)
        alphas = torch.linspace(0, 1, steps)
        gradients = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (data - baseline)
            interpolated.requires_grad = True
            output = model(interpolated)
            output[:, 1].backward()
            gradients.append(interpolated.grad)
        
        integrated = torch.stack(gradients).mean(dim=0) * (data - baseline)
        return {'integrated_gradients': integrated.cpu().numpy()}
    
    def _get_top_features(self, explanation: Dict, k: int = 10) -> List[Dict]:
        if 'shap_values' in explanation:
            values = explanation['shap_values']
            if isinstance(values, np.ndarray):
                indices = np.argsort(np.abs(values).mean(axis=0))[-k:][::-1]
                return [
                    {
                        'feature': self.feature_names[i] if i < len(self.feature_names) else f'Feature_{i}',
                        'importance': float(values.mean(axis=0)[i]),
                        'index': int(i)
                    }
                    for i in indices
                ]
        return []
    
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == "explain":
            model = message.payload['model']
            data = message.payload['data']
            prediction = message.payload['prediction']
            
            explanation = self.generate_explanation(model, data, prediction)
            
            response = AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type="explanation_result",
                payload=explanation,
                timestamp=message.timestamp
            )
            return response
        return None


class OptimizationAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, AgentType.OPTIMIZER, config)
        self.optimization_history = []
        self.hyperparameter_space = config.get('hyperparameter_space', {})
        self.optimization_method = config.get('method', 'bayesian')
    
    def optimize_hyperparameters(self, model, train_data, val_data, metric='f1') -> Dict:
        if self.optimization_method == 'bayesian':
            return self._bayesian_optimization(model, train_data, val_data, metric)
        elif self.optimization_method == 'grid_search':
            return self._grid_search(model, train_data, val_data, metric)
        else:
            return self._random_search(model, train_data, val_data, metric)
    
    def _bayesian_optimization(self, model, train_data, val_data, metric: str) -> Dict:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
        
        space = []
        param_names = []
        
        for name, bounds in self.hyperparameter_space.items():
            if isinstance(bounds, tuple) and len(bounds) == 2:
                if isinstance(bounds[0], int):
                    space.append(Integer(bounds[0], bounds[1], name=name))
                else:
                    space.append(Real(bounds[0], bounds[1], name=name))
                param_names.append(name)
        
        def objective(params):
            params_dict = dict(zip(param_names, params))
            model.set_hyperparameters(params_dict)
            score = self._evaluate_model(model, train_data, val_data, metric)
            return -score
        
        result = gp_minimize(objective, space, n_calls=20, random_state=42)
        best_params = dict(zip(param_names, result.x))
        
        return {
            'best_params': best_params,
            'best_score': -result.fun,
            'optimization_history': result.func_vals.tolist()
        }
    
    def _grid_search(self, model, train_data, val_data, metric: str) -> Dict:
        from itertools import product
        
        param_grid = {}
        for name, values in self.hyperparameter_space.items():
            if isinstance(values, list):
                param_grid[name] = values
        
        best_score = -float('inf')
        best_params = {}
        
        for params in product(*param_grid.values()):
            params_dict = dict(zip(param_grid.keys(), params))
            model.set_hyperparameters(params_dict)
            score = self._evaluate_model(model, train_data, val_data, metric)
            
            if score > best_score:
                best_score = score
                best_params = params_dict
        
        return {'best_params': best_params, 'best_score': best_score}
    
    def _random_search(self, model, train_data, val_data, metric: str, n_iter: int = 20) -> Dict:
        import random
        
        best_score = -float('inf')
        best_params = {}
        
        for _ in range(n_iter):
            params_dict = {}
            for name, bounds in self.hyperparameter_space.items():
                if isinstance(bounds, tuple):
                    if isinstance(bounds[0], int):
                        params_dict[name] = random.randint(bounds[0], bounds[1])
                    else:
                        params_dict[name] = random.uniform(bounds[0], bounds[1])
                elif isinstance(bounds, list):
                    params_dict[name] = random.choice(bounds)
            
            model.set_hyperparameters(params_dict)
            score = self._evaluate_model(model, train_data, val_data, metric)
            
            if score > best_score:
                best_score = score
                best_params = params_dict
        
        return {'best_params': best_params, 'best_score': best_score}
    
    def _evaluate_model(self, model, train_data, val_data, metric: str) -> float:
        model.train()
        train_loss = self._train_epoch(model, train_data)
        
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_data:
                outputs = model(batch['data'])
                val_predictions.append(outputs.argmax(dim=1))
                val_targets.append(batch['target'])
        
        from sklearn.metrics import f1_score, accuracy_score
        
        predictions = torch.cat(val_predictions).cpu().numpy()
        targets = torch.cat(val_targets).cpu().numpy()
        
        if metric == 'f1':
            return f1_score(targets, predictions)
        else:
            return accuracy_score(targets, predictions)
    
    def _train_epoch(self, model, train_data):
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for batch in train_data:
            optimizer.zero_grad()
            outputs = model(batch['data'])
            loss = criterion(outputs, batch['target'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_data)
    
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == "optimize":
            model = message.payload['model']
            train_data = message.payload['train_data']
            val_data = message.payload['val_data']
            metric = message.payload.get('metric', 'f1')
            
            optimization_result = self.optimize_hyperparameters(model, train_data, val_data, metric)
            
            response = AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type="optimization_result",
                payload=optimization_result,
                timestamp=message.timestamp
            )
            return response
        return None


class CoordinatorAgent(BaseAgent):
    def __init__(self, agent_id: str, agents: List[BaseAgent], config: Dict):
        super().__init__(agent_id, AgentType.COORDINATOR, config)
        self.agents = {agent.agent_id: agent for agent in agents}
        self.workflow = config.get('workflow', [])
        self.consensus_threshold = config.get('consensus_threshold', 0.6)
    
    async def coordinate_detection(self, data: torch.Tensor) -> Dict:
        detector_agents = [a for a in self.agents.values() if a.agent_type == AgentType.DETECTOR]
        
        predictions = []
        for agent in detector_agents:
            is_fraud, result = agent.forward(data)
            predictions.append({
                'agent_id': agent.agent_id,
                'prediction': is_fraud.item(),
                'confidence': result['confidence'].item(),
                'result': result
            })
        
        consensus = self._compute_consensus(predictions)
        
        if consensus['is_fraud']:
            validator_agents = [a for a in self.agents.values() if a.agent_type == AgentType.VALIDATOR]
            
            validation_results = []
            for agent in validator_agents:
                message = AgentMessage(
                    sender=self.agent_id,
                    receiver=agent.agent_id,
                    message_type="validate",
                    payload={
                        'prediction': consensus,
                        'context': {'historical': [p['confidence'] for p in predictions]}
                    },
                    timestamp=0.0
                )
                validation_result = agent.process_message(message)
                if validation_result:
                    validation_results.append(validation_result.payload)
            
            if validation_results:
                consensus['validation'] = validation_results
                consensus['is_validated'] = any(v['is_valid'] for v in validation_results)
        
        if consensus.get('is_validated', False):
            explainer_agents = [a for a in self.agents.values() if a.agent_type == AgentType.EXPLAINER]
            
            if explainer_agents:
                agent = explainer_agents[0]
                message = AgentMessage(
                    sender=self.agent_id,
                    receiver=agent.agent_id,
                    message_type="explain",
                    payload={
                        'model': detector_agents[0].model,
                        'data': data,
                        'prediction': consensus
                    },
                    timestamp=0.0
                )
                explanation = agent.process_message(message)
                if explanation:
                    consensus['explanation'] = explanation.payload
        
        return consensus
    
    def _compute_consensus(self, predictions: List[Dict]) -> Dict:
        fraud_votes = sum(1 for p in predictions if p['prediction'] == 1)
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        
        is_fraud = (fraud_votes / len(predictions)) >= self.consensus_threshold
        
        return {
            'is_fraud': int(is_fraud),
            'confidence': float(avg_confidence),
            'votes': fraud_votes,
            'total_agents': len(predictions),
            'consensus_ratio': fraud_votes / len(predictions) if predictions else 0.0
        }
    
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == "coordinate":
            data = message.payload['data']
            result = asyncio.run(self.coordinate_detection(data))
            
            response = AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type="coordination_result",
                payload=result,
                timestamp=message.timestamp
            )
            return response
        return None


class MultiAgentSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.agents = {}
        self.coordinator = None
        self.message_bus = deque(maxlen=10000)
    
    def add_agent(self, agent: BaseAgent):
        self.agents[agent.agent_id] = agent
    
    def initialize_coordinator(self):
        detector_agents = [a for a in self.agents.values() if a.agent_type == AgentType.DETECTOR]
        validator_agents = [a for a in self.agents.values() if a.agent_type == AgentType.VALIDATOR]
        explainer_agents = [a for a in self.agents.values() if a.agent_type == AgentType.EXPLAINER]
        
        all_agents = detector_agents + validator_agents + explainer_agents
        
        self.coordinator = CoordinatorAgent(
            "coordinator_0",
            all_agents,
            self.config.get('coordinator', {})
        )
    
    async def process_transaction(self, data: torch.Tensor) -> Dict:
        if self.coordinator is None:
            self.initialize_coordinator()
        
        message = AgentMessage(
            sender="system",
            receiver=self.coordinator.agent_id,
            message_type="coordinate",
            payload={'data': data},
            timestamp=0.0
        )
        
        result = self.coordinator.process_message(message)
        if result:
            return result.payload
        return {}
