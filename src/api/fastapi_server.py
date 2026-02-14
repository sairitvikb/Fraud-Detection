from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
import numpy as np
import redis
import json
import logging
from datetime import datetime
import asyncio
import os
from pathlib import Path

logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    redis_available = True
except:
    redis_available = False
    logger.warning("Redis not available, caching disabled")

model = None
explainer = None
agent_system = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransactionRequest(BaseModel):
    transaction_id: str
    pixel_values: Optional[List[List[float]]] = None
    node_features: Optional[List[List[float]]] = None
    edge_index: Optional[List[List[int]]] = None
    feature_vector: Optional[List[float]] = None
    metadata: Optional[Dict] = None


class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    confidence: float
    explanation: Optional[Dict] = None
    processing_time_ms: float


@app.on_event("startup")
async def startup_event():
    global model, explainer, agent_system
    logger.info("Loading models...")
    
    try:
        from models.advanced_ensemble import AdaptiveWeightedEnsemble
        from models.vision_transformer import VisionTransformerFraudDetector
        from models.graph_neural_network import GATFraudDetector
        
        model_path = 'models/best_model.pt'
        if os.path.exists(model_path):
            vit_model = VisionTransformerFraudDetector(num_classes=2)
            gnn_model = GATFraudDetector(input_dim=10, hidden_dim=128, num_layers=3, num_heads=8)
            
            model = AdaptiveWeightedEnsemble(
                models=[vit_model, gnn_model],
                num_classes=2
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}. Using untrained model.")
            vit_model = VisionTransformerFraudDetector(num_classes=2)
            gnn_model = GATFraudDetector(input_dim=10, hidden_dim=128)
            model = AdaptiveWeightedEnsemble(models=[vit_model, gnn_model], num_classes=2)
            model.to(device)
            model.eval()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
    
    try:
        from explainability.shap_explainer import SHAPExplainer
        if model:
            explainer = SHAPExplainer(model)
    except Exception as e:
        logger.warning(f"Explainer not available: {e}")
        explainer = None
    
    logger.info("Models loaded")


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(request: TransactionRequest, background_tasks: BackgroundTasks):
    start_time = datetime.now()
    
    if redis_available:
        cache_key = f"prediction:{request.transaction_id}"
        cached = redis_client.get(cache_key)
        
        if cached:
            result = json.loads(cached)
            result['cached'] = True
            return PredictionResponse(**result)
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        inputs = {}
        
        if request.pixel_values:
            pixel_tensor = torch.tensor([request.pixel_values], dtype=torch.float32).to(device)
            if len(pixel_tensor.shape) == 3:
                pixel_tensor = pixel_tensor.unsqueeze(0)
            inputs['pixel_values'] = pixel_tensor
        elif request.node_features and request.edge_index:
            node_tensor = torch.tensor([request.node_features], dtype=torch.float32).to(device)
            edge_tensor = torch.tensor(request.edge_index, dtype=torch.long).to(device)
            inputs['x'] = node_tensor
            inputs['edge_index'] = edge_tensor
        elif request.feature_vector:
            feature_tensor = torch.tensor([request.feature_vector], dtype=torch.float32).to(device)
            inputs['data'] = feature_tensor
        else:
            raise HTTPException(status_code=400, detail="Invalid input format. Provide pixel_values, (node_features + edge_index), or feature_vector")
        
        with torch.no_grad():
            outputs = model(**inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            fraud_prob = probs[0, 1].item()
            is_fraud = fraud_prob > 0.5
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = {
            'transaction_id': request.transaction_id,
            'is_fraud': bool(is_fraud),
            'confidence': float(fraud_prob),
            'processing_time_ms': processing_time
        }
        
        if explainer and request.feature_vector:
            try:
                explanation = explainer.explain(torch.tensor([request.feature_vector], dtype=torch.float32))
                result['explanation'] = explanation
            except:
                pass
        
        if redis_available:
            redis_client.setex(cache_key, 3600, json.dumps(result))
        
        background_tasks.add_task(log_prediction, request.transaction_id, result)
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(requests: List[TransactionRequest]):
    results = []
    
    for request in requests:
        try:
            response = await predict_fraud(request, BackgroundTasks())
            results.append(response.dict())
        except Exception as e:
            results.append({
                'transaction_id': request.transaction_id,
                'error': str(e)
            })
    
    return {'results': results, 'total': len(results)}


@app.get("/explain/{transaction_id}")
async def explain_prediction(transaction_id: str):
    if explainer:
        explanation = explainer.explain(transaction_id)
        return {'transaction_id': transaction_id, 'explanation': explanation}
    return {'error': 'Explainer not available'}


@app.get("/health")
async def health_check():
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'redis_connected': redis_available,
        'device': str(device)
    }


async def log_prediction(transaction_id: str, result: Dict):
    if redis_available:
        log_entry = {
            'transaction_id': transaction_id,
            'timestamp': datetime.now().isoformat(),
            'result': result
        }
        redis_client.lpush('prediction_logs', json.dumps(log_entry))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
