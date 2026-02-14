from kafka import KafkaConsumer
from kafka.errors import KafkaError
import json
import logging
import torch
from typing import Dict, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


class FraudDetectionKafkaConsumer:
    def __init__(self,
                 bootstrap_servers: str = "localhost:9092",
                 topic: str = "transactions",
                 consumer_group: str = "fraud_detection",
                 batch_size: int = 1000,
                 max_poll_records: int = 500,
                 model=None,
                 processor=None):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer_group = consumer_group
        self.batch_size = batch_size
        self.max_poll_records = max_poll_records
        self.model = model
        self.processor = processor
        
        self.consumer = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.processed_count = 0
        self.fraud_count = 0
        
    def initialize(self):
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.consumer_group,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                max_poll_records=self.max_poll_records,
                consumer_timeout_ms=1000
            )
            logger.info(f"Kafka consumer initialized for topic: {self.topic}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise
    
    def process_batch(self, messages: List[Dict]) -> List[Dict]:
        if self.processor is None:
            return messages
        
        results = []
        batch_data = []
        
        for msg in messages:
            try:
                processed = self.processor.preprocess(msg.value)
                batch_data.append(processed)
            except Exception as e:
                logger.error(f"Error preprocessing message: {e}")
                continue
        
        if batch_data and self.model is not None:
            try:
                predictions = self.processor.predict_batch(self.model, batch_data)
                
                for msg, pred in zip(messages, predictions):
                    result = {
                        'transaction_id': msg.value.get('transaction_id'),
                        'prediction': pred['is_fraud'],
                        'confidence': pred['confidence'],
                        'timestamp': time.time()
                    }
                    results.append(result)
                    
                    if pred['is_fraud']:
                        self.fraud_count += 1
                    
                    self.processed_count += 1
            except Exception as e:
                logger.error(f"Error in batch prediction: {e}")
        
        return results
    
    def start_consuming(self):
        if self.consumer is None:
            self.initialize()
        
        self.running = True
        logger.info("Starting Kafka consumer...")
        
        batch = []
        last_batch_time = time.time()
        
        try:
            while self.running:
                message_pack = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        batch.append(message)
                        
                        if len(batch) >= self.batch_size:
                            future = self.executor.submit(self.process_batch, batch)
                            batch = []
                            last_batch_time = time.time()
                
                if len(batch) > 0 and (time.time() - last_batch_time) > 5.0:
                    future = self.executor.submit(self.process_batch, batch)
                    batch = []
                    last_batch_time = time.time()
                
                if self.processed_count % 10000 == 0 and self.processed_count > 0:
                    logger.info(f"Processed {self.processed_count} transactions, "
                              f"detected {self.fraud_count} fraud cases")
        
        except KafkaError as e:
            logger.error(f"Kafka error: {e}")
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        finally:
            self.stop()
    
    def stop(self):
        self.running = False
        if self.consumer:
            self.consumer.close()
        self.executor.shutdown(wait=True)
        logger.info(f"Consumer stopped. Total processed: {self.processed_count}, "
                   f"Fraud detected: {self.fraud_count}")
