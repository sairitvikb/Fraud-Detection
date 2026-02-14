import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np
import logging
from sklearn.model_selection import ParameterGrid
import optuna
from optuna.trial import TrialState

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    def __init__(self, model_class, config: Dict):
        self.model_class = model_class
        self.config = config
        self.best_params = None
        self.best_score = -float('inf')
        self.optimization_history = []
    
    def grid_search(self, param_grid: Dict, train_loader, val_loader, 
                   device: str = "cuda", num_epochs: int = 5) -> Dict:
        best_params = None
        best_score = -float('inf')
        results = []
        
        for params in ParameterGrid(param_grid):
            logger.info(f"Testing parameters: {params}")
            
            model = self.model_class(**params)
            model.to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=params.get('lr', 1e-4))
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(num_epochs):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch['data'].to(device))
                    loss = criterion(outputs, batch['target'].to(device))
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(batch['data'].to(device))
                    pred = outputs.argmax(dim=1)
                    correct += pred.eq(batch['target'].to(device)).sum().item()
                    total += batch['target'].size(0)
            
            score = 100. * correct / total
            
            results.append({
                'params': params,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = params
        
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def bayesian_optimization(self, search_space: Dict, train_loader, val_loader,
                            device: str = "cuda", n_trials: int = 50) -> Dict:
        def objective(trial):
            params = {}
            for name, space in search_space.items():
                if isinstance(space, dict):
                    if space['type'] == 'float':
                        params[name] = trial.suggest_float(
                            name, space['low'], space['high'], log=space.get('log', False)
                        )
                    elif space['type'] == 'int':
                        params[name] = trial.suggest_int(
                            name, space['low'], space['high'], log=space.get('log', False)
                        )
                    elif space['type'] == 'categorical':
                        params[name] = trial.suggest_categorical(name, space['choices'])
                elif isinstance(space, list):
                    params[name] = trial.suggest_categorical(name, space)
            
            model = self.model_class(**params)
            model.to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=params.get('lr', 1e-4))
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(3):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch['data'].to(device))
                    loss = criterion(outputs, batch['target'].to(device))
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(batch['data'].to(device))
                    pred = outputs.argmax(dim=1)
                    correct += pred.eq(batch['target'].to(device)).sum().item()
                    total += batch['target'].size(0)
            
            score = 100. * correct / total
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def evolutionary_optimization(self, search_space: Dict, train_loader, val_loader,
                                 device: str = "cuda", population_size: int = 20,
                                 generations: int = 10) -> Dict:
        def create_individual():
            individual = {}
            for name, space in search_space.items():
                if isinstance(space, dict):
                    if space['type'] == 'float':
                        individual[name] = np.random.uniform(space['low'], space['high'])
                    elif space['type'] == 'int':
                        individual[name] = np.random.randint(space['low'], space['high'] + 1)
                    elif space['type'] == 'categorical':
                        individual[name] = np.random.choice(space['choices'])
                elif isinstance(space, list):
                    individual[name] = np.random.choice(space)
            return individual
        
        def evaluate_individual(individual):
            try:
                model = self.model_class(**individual)
                model.to(device)
                
                optimizer = torch.optim.AdamW(model.parameters(), lr=individual.get('lr', 1e-4))
                criterion = nn.CrossEntropyLoss()
                
                for epoch in range(2):
                    model.train()
                    for batch in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch['data'].to(device))
                        loss = criterion(outputs, batch['target'].to(device))
                        loss.backward()
                        optimizer.step()
                
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        outputs = model(batch['data'].to(device))
                        pred = outputs.argmax(dim=1)
                        correct += pred.eq(batch['target'].to(device)).sum().item()
                        total += batch['target'].size(0)
                
                return 100. * correct / total
            except:
                return -float('inf')
        
        def crossover(parent1, parent2):
            child = {}
            for key in parent1:
                if np.random.random() < 0.5:
                    child[key] = parent1[key]
                else:
                    child[key] = parent2[key]
            return child
        
        def mutate(individual, mutation_rate=0.1):
            mutated = individual.copy()
            for name, space in search_space.items():
                if np.random.random() < mutation_rate:
                    if isinstance(space, dict):
                        if space['type'] == 'float':
                            mutated[name] = np.random.uniform(space['low'], space['high'])
                        elif space['type'] == 'int':
                            mutated[name] = np.random.randint(space['low'], space['high'] + 1)
                        elif space['type'] == 'categorical':
                            mutated[name] = np.random.choice(space['choices'])
                    elif isinstance(space, list):
                        mutated[name] = np.random.choice(space)
            return mutated
        
        population = [create_individual() for _ in range(population_size)]
        
        for generation in range(generations):
            fitness_scores = [evaluate_individual(ind) for ind in population]
            sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
            
            elite_size = population_size // 4
            elite = [ind for ind, score in sorted_pop[:elite_size]]
            
            new_population = elite.copy()
            
            while len(new_population) < population_size:
                parent1, parent2 = np.random.choice(len(elite), 2, replace=False)
                child = crossover(elite[parent1], elite[parent2])
                child = mutate(child)
                new_population.append(child)
            
            population = new_population
            best_score = max(fitness_scores)
            logger.info(f"Generation {generation+1}: Best score = {best_score:.4f}")
        
        final_scores = [evaluate_individual(ind) for ind in population]
        best_idx = np.argmax(final_scores)
        
        self.best_params = population[best_idx]
        self.best_score = final_scores[best_idx]
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score
        }
