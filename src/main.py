import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List
import argparse
import logging
import yaml
import os

sys.path.append(str(Path(__file__).parent))

from models.vision_transformer import VisionTransformerFraudDetector, MultiScaleVisionTransformer
from models.graph_neural_network import GATFraudDetector, GCNFraudDetector, GraphSAGEFraudDetector
from models.ensemble import MultimodalEnsemble
from models.advanced_agents import MultiAgentSystem, FraudDetectorAgent, ValidationAgent, ExplanationAgent
from models.advanced_ensemble import AdaptiveWeightedEnsemble, StackingEnsemble, DynamicEnsemble, BayesianEnsemble
from models.transformer_models import TransformerEncoderFraudDetector, MultiModalTransformer, TemporalTransformer
from training.sophisticated_training import AdvancedTrainer
from optimization.hyperparameter_tuning import HyperparameterOptimizer
from explainability.shap_explainer import SHAPExplainer
from data.graph_builder import TransactionGraphBuilder, TemporalGraphBuilder
from data.data_loader import DataManager
from evaluation.advanced_metrics import AdvancedEvaluator
from evaluation.sophisticated_metrics import ComprehensiveEvaluator
from streaming.kafka_consumer import FraudDetectionKafkaConsumer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_default_config() -> Dict:
    return {
        'models': {
            'vision_transformer': {
                'model_name': 'google/vit-base-patch16-224',
                'num_classes': 2
            },
            'graph_neural_network': {
                'hidden_dim': 128,
                'num_layers': 3,
                'num_heads': 8
            },
            'ensemble': {
                'fusion_method': 'weighted'
            }
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'use_mixed_precision': True,
            'gradient_accumulation_steps': 1,
            'use_ema': True,
            'ema_decay': 0.999,
            'mixup_prob': 0.2,
            'cutmix_prob': 0.1,
            'label_smoothing': 0.1,
            'gradient_clip': 1.0
        },
        'optimizer': {
            'type': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 0.01
        },
        'scheduler': {
            'type': 'cosine',
            'T_max': 10
        },
        'loss': {
            'type': 'focal',
            'alpha': 1.0,
            'gamma': 2.0
        },
        'api': {
            'host': '0.0.0.0',
            'port': 8000
        },
        'streaming': {
            'kafka': {
                'bootstrap_servers': 'localhost:9092',
                'topic': 'transactions',
                'consumer_group': 'fraud_detection'
            }
        }
    }


def train_models(config: Dict, data_path: str):
    logger.info("=" * 80)
    logger.info("TRAINING FRAUD DETECTION MODELS")
    logger.info("=" * 80)
    
    data_manager = DataManager(config)
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.info("Creating synthetic data for demonstration...")
        df = create_synthetic_data(10000)
        data_path = '/tmp/synthetic_transactions.csv'
        df.to_csv(data_path, index=False)
        logger.info(f"Synthetic data saved to {data_path}")
    
    df = data_manager.load_data(data_path)
    train_df, val_df, test_df = data_manager.preprocess_data(df)
    
    train_loader, val_loader, test_loader = data_manager.create_dataloaders(
        train_df, val_df, test_df,
        batch_size=config['training']['batch_size']
    )
    
    input_dim = len(data_manager.feature_cols) if data_manager.feature_cols else 10
    
    vit_model = VisionTransformerFraudDetector(
        model_name=config['models']['vision_transformer']['model_name'],
        num_classes=2
    )
    
    gnn_model = GATFraudDetector(
        input_dim=input_dim,
        hidden_dim=config['models']['graph_neural_network']['hidden_dim'],
        num_layers=config['models']['graph_neural_network']['num_layers'],
        num_heads=config['models']['graph_neural_network']['num_heads']
    )
    
    ensemble = AdaptiveWeightedEnsemble(
        models=[vit_model, gnn_model],
        num_classes=2,
        use_meta_learner=True
    )
    
    trainer = AdvancedTrainer(
        model=ensemble,
        config=config['training'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader, config['training']['num_epochs'])
    
    logger.info("Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    logger.info(f"Test Metrics: {test_metrics}")
    
    evaluator = ComprehensiveEvaluator()
    all_preds = []
    all_targets = []
    all_probs = []
    
    trainer.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs = trainer._prepare_inputs(batch)
            targets = batch['target'].to(trainer.device)
            outputs = trainer.model(**inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    comprehensive_metrics = evaluator.evaluate_comprehensive(
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs)
    )
    logger.info(f"Comprehensive Metrics: {comprehensive_metrics}")
    
    model_save_path = 'models/best_model.pt'
    os.makedirs('models', exist_ok=True)
    torch.save(trainer.model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")


def create_synthetic_data(n_samples: int = 10000) -> pd.DataFrame:
    np.random.seed(42)
    
    data = {
        'transaction_id': [f'txn_{i}' for i in range(n_samples)],
        'amount': np.random.lognormal(mean=3, sigma=1, size=n_samples),
        'merchant_id': np.random.randint(1, 100, n_samples),
        'account_id': np.random.randint(1, 1000, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'location_lat': np.random.uniform(-90, 90, n_samples),
        'location_lon': np.random.uniform(-180, 180, n_samples),
        'transaction_type': np.random.choice(['purchase', 'withdrawal', 'transfer'], n_samples),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    fraud_prob = 0.05 + 0.1 * (df['amount'] > df['amount'].quantile(0.9)).astype(int)
    fraud_prob += 0.1 * (df['hour'] < 6).astype(int)
    fraud_prob += 0.05 * (df['location_lat'].abs() > 60).astype(int)
    
    df['is_fraud'] = (np.random.random(n_samples) < fraud_prob).astype(int)
    
    return df


def predict_fraud(config: Dict, data_path: str, model_path: str = 'models/best_model.pt'):
    logger.info("Running fraud prediction...")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}. Please train a model first.")
        return
    
    data_manager = DataManager(config)
    df = data_manager.load_data(data_path)
    
    model = AdaptiveWeightedEnsemble(
        models=[
            VisionTransformerFraudDetector(num_classes=2),
            GATFraudDetector(input_dim=len(data_manager.feature_cols) if data_manager.feature_cols else 10, hidden_dim=128)
        ],
        num_classes=2
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    dataset = data_manager.create_dataloaders(df, df, df, batch_size=32)[0].dataset
    
    predictions = []
    with torch.no_grad():
        for sample in dataset:
            inputs = {'data': sample['data'].unsqueeze(0)}
            output = model(inputs)
            if isinstance(output, tuple):
                output = output[0]
            prob = torch.softmax(output, dim=1)[0, 1].item()
            predictions.append({
                'transaction_id': sample['transaction_id'],
                'fraud_probability': prob,
                'is_fraud': prob > 0.5
            })
    
    results_df = pd.DataFrame(predictions)
    output_path = 'predictions.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


def start_streaming(config: Dict):
    logger.info("Starting Kafka streaming consumer...")
    
    try:
        consumer = FraudDetectionKafkaConsumer(
            bootstrap_servers=config['streaming']['kafka']['bootstrap_servers'],
            topic=config['streaming']['kafka']['topic'],
            consumer_group=config['streaming']['kafka']['consumer_group']
        )
        consumer.start_consuming()
    except Exception as e:
        logger.error(f"Error starting streaming: {e}")
        logger.info("Kafka not available. Skipping streaming mode.")


def start_api(config: Dict):
    logger.info("Starting FastAPI server...")
    from api.fastapi_server import app
    import uvicorn
    
    uvicorn.run(app, host=config['api']['host'], port=config['api']['port'])


def main():
    parser = argparse.ArgumentParser(description='Fraud Detection Platform')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'stream', 'api'], default='train')
    parser.add_argument('--data', type=str, default='data/transactions.csv')
    parser.add_argument('--model', type=str, default='models/best_model.pt')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    logger.info("=" * 80)
    logger.info("FRAUD DETECTION & CUSTOMER INTELLIGENCE PLATFORM")
    logger.info("=" * 80)
    
    if args.mode == 'train':
        train_models(config, args.data)
    elif args.mode == 'predict':
        predict_fraud(config, args.data, args.model)
    elif args.mode == 'stream':
        start_streaming(config)
    elif args.mode == 'api':
        start_api(config)


if __name__ == '__main__':
    main()
