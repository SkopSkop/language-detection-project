import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str = "xlm-roberta-base"
    num_labels: int = 20
    max_length: int = 512
    dropout_rate: float = 0.1
    
@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_steps: int = 100
    
@dataclass
class PathConfig:
    data_dir: str = "./data"
    model_dir: str = "./models"
    log_dir: str = "./logs"
    experiment_dir: str = "./experiments"