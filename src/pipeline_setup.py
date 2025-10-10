# src/pipeline_setup.py
"""
Core pipeline setup and configuration management for the synthetic data generation pipeline.
"""

import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None  # e.g., ["q_proj","v_proj"] for LLaMA-style

@dataclass
class GenConfig:
    model_name: str = "meta-llama/Llama-3.2-3B" #"meta-llama/Llama-2-7b-hf" #"meta-llama/Llama-3.2-3B" 
    device: str = "auto"
    max_new_tokens: int = 128
    temperature: float = 0.8   # τ_g in the papers
    top_p: float = 0.95
    do_sample: bool = True
    batch_size: int = 4
    seed: int = 42
    lora: Optional[LoRAConfig] = None  # if set, LoRA branch is enabled

@dataclass
class DataConfig:
    dataset: str = "ag_news"  # "ag_news" or "imdb"
    fewshot_k_per_class: int = 0  # 0 => zero-shot; >0 => few-shot (# per class)
    limit_train: Optional[int] = None  # optional cap for training items
    limit_seed: Optional[int] = None   # optional cap for seed sampling

@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    gen: GenConfig = field(default_factory=GenConfig)
    out_dir: str = "outputs"
    mode: str = "zero_shot"  # "zero_shot" | "few_shot" | "lora_finetune"
    synth_per_class: int = 500
    eval_dcscore: bool = True
    eval_basic: bool = True  # distinct-n, self-BLEU



class Config:
    """Central configuration management"""
    
    def __init__(self):
        # Get the project root (parent of src directory)
        self.src_dir = Path(__file__).parent
        self.base_dir = self.src_dir.parent
        
        # Main directories
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        self.reports_dir = self.base_dir / "reports"
        
        # Ensure directories exist
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "synthetic").mkdir(exist_ok=True)
        (self.data_dir / "uploads").mkdir(exist_ok=True)
        (self.reports_dir / "validation").mkdir(exist_ok=True)
        
        # # Model configurations
        # self.model_configs = {
        #     "generator": {
        #         "model_name": "distilgpt2",
        #         "max_length": 100,
        #         "num_return_sequences": 5,
        #         "temperature": 0.8,
        #         "top_k": 50,
        #         "top_p": 0.9
        #     },
        #     "embeddings": {
        #         "model_name": "all-MiniLM-L6-v2"
        #     },
        #     "validation_thresholds": {
        #         "min_uniqueness_ratio": 0.7,
        #         "max_exact_duplicates": 0.05,
        #         "min_avg_perplexity": 5.0,
        #         "max_avg_perplexity": 100.0,
        #         "min_embedding_similarity": 0.3
        #     }
        # }


        ## meta-llama/Llama-2-7b-hf ## - GPU
        
        # Model configurations
        self.model_configs = {
            "generator": {
                "model_name": "meta-llama/Llama-3.2-3B",   # <-- swap here
                "max_length": 512,                         # can go longer than GPT2
                "num_return_sequences": 5,
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.9
            },
            "embeddings": {
                "model_name": "all-MiniLM-L6-v2"
            },
            "validation_thresholds": {
                "min_uniqueness_ratio": 0.7,
                "max_exact_duplicates": 0.05,
                "min_avg_perplexity": 5.0,
                "max_avg_perplexity": 100.0,
                "min_embedding_similarity": 0.3
            }
        }

        
        # PEFT/LoRA configuration
        self.lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["c_attn", "c_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
        
        # Training configuration
        self.training_config = {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False
        }
    
    def save_config(self, config_path: Optional[Path] = None):
        """Save configuration to JSON file"""
        if config_path is None:
            config_dir = self.base_dir / "config"
            config_dir.mkdir(exist_ok=True)
            config_path = config_dir / "config.json"
        
        config_dict = {
            "model_configs": self.model_configs,
            "lora_config": self.lora_config,
            "training_config": self.training_config
        }
        
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to {config_path}")

class MetadataDB:
    """SQLite database for tracking pipeline metadata"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    type TEXT NOT NULL, -- 'real' or 'synthetic'
                    size INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_path TEXT NOT NULL,
                    metadata JSON
                );
                
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    parameters JSON NOT NULL,
                    metrics JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    mlflow_run_id TEXT,
                    status TEXT DEFAULT 'running'
                );
                
                CREATE TABLE IF NOT EXISTS validations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER NOT NULL,
                    validation_type TEXT NOT NULL,
                    result JSON NOT NULL,
                    passed BOOLEAN NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dataset_id) REFERENCES datasets (id)
                );
            """)
    
    def log_dataset(self, name: str, version: str, type_: str, 
                   size: int, file_path: str, metadata: Dict = None):
        """Log a new dataset"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO datasets (name, version, type, size, file_path, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, version, type_, size, file_path, json.dumps(metadata or {})))
            return cursor.lastrowid
    
    def save_dataset(self, name: str, version: str, dataset_type: str, 
                    file_path: str, size: int, metadata: Dict = None):
        """Save dataset (alias for log_dataset for compatibility)"""
        return self.log_dataset(name, version, dataset_type, size, file_path, metadata)
    
    def log_experiment(self, name: str, model_name: str, 
                      parameters: Dict, mlflow_run_id: str = None):
        """Log a new experiment"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (name, model_name, parameters, mlflow_run_id)
                VALUES (?, ?, ?, ?)
            """, (name, model_name, json.dumps(parameters), mlflow_run_id))
            return cursor.lastrowid
    
    def log_validation(self, dataset_id: int, validation_type: str, 
                      result: Dict, passed: bool):
        """Log validation results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO validations (dataset_id, validation_type, result, passed)
                VALUES (?, ?, ?, ?)
            """, (dataset_id, validation_type, json.dumps(result), passed))
            return cursor.lastrowid

import logging
from typing import Union
from pathlib import Path

def setup_logging(logs_dir: Path, log_level: Union[str, int] = "INFO") -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Coerce level
    if isinstance(log_level, int):
        level = log_level
    else:
        level = getattr(logging, str(log_level).upper(), logging.INFO)

    # Use a named logger so we don't clobber Prefect/root handlers
    logger_name = "synthetic_pipeline"
    lg = logging.getLogger(logger_name)
    lg.setLevel(level)
    lg.propagate = True  # let messages bubble up to Prefect/root if configured

    # Formatter
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add handlers only once
    have_file = any(isinstance(h, logging.FileHandler) for h in lg.handlers)
    have_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                       for h in lg.handlers)

    if not have_file:
        fh = logging.FileHandler(str(logs_dir / "pipeline.log"))
        fh.setLevel(level)
        fh.setFormatter(fmt)
        lg.addHandler(fh)

    if not have_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        lg.addHandler(ch)

    return lg
    


def initialize_project():
    """Initialize the complete project setup"""
    print("Initializing Synthetic Language Data Pipeline...")
    
    # Initialize configuration
    config = Config()
    config.save_config()
    
    # Setup database
    db = MetadataDB(config.data_dir / "pipeline.db")
    
    # Setup logging
    logger = setup_logging(config.logs_dir)
    logger.info("Pipeline initialization completed successfully!")
    
    # Setup MLflow tracking
    try:
        import mlflow
        mlflow_dir = config.base_dir / "mlruns"
        mlflow_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        logger.info("MLflow tracking configured")
    except ImportError:
        logger.warning("MLflow not available")
    
    print(" Project initialization complete!")
    return config, db, logger

if __name__ == "__main__":
    # Initialize project when run directly
    config, db, logger = initialize_project()
    
    print(f" Project structure:")
    print(f"   Base directory: {config.base_dir}")
    print(f"   Data directory: {config.data_dir}")
    print(f"   Models directory: {config.models_dir}")
    print(f"   Logs directory: {config.logs_dir}")
    print(f"   Database: {config.data_dir / 'pipeline.db'}")