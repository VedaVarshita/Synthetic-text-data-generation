# # pipelines/prefect/flows.py
# """
# Prefect flows for orchestrating the synthetic data generation pipeline.
# Includes end-to-end pipeline, scheduled generation, and monitoring flows.
# """

# import os
# import json
# import logging
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# from datetime import datetime, timedelta
# import pandas as pd

# from prefect import flow, task, get_run_logger
# from prefect.task_runners import SequentialTaskRunner, ConcurrentTaskRunner
# from prefect.blocks.system import Secret
# from prefect.server.schemas.schedules import CronSchedule, IntervalSchedule
# import mlflow

# # Setup project paths
# PROJECT_ROOT = Path(__file__).parent.parent.parent
# import sys
# sys.path.append(str(PROJECT_ROOT))

# from src.pipeline_setup import Config, MetadataDB, setup_logging
# from src.ingestion import DataIngestion
# from src.generator import SyntheticTextGenerator
# from src.validate import ValidationPipeline

# # Initialize global configuration
# config = Config()
# metadata_db = MetadataDB(config.data_dir / "pipeline.db")

# # Prefect tasks
# @task(name="setup_environment", description="Setup pipeline environment and check dependencies")
# def setup_environment_task() -> Dict[str, Any]:
#     """Setup and validate pipeline environment"""
#     logger = get_run_logger()
#     logger.info("Setting up pipeline environment...")
    
#     try:
#         # Check required directories
#         for dir_path in [config.data_dir, config.models_dir, config.logs_dir]:
#             dir_path.mkdir(exist_ok=True)
        
#         # Check MLflow setup
#         mlflow.set_tracking_uri(f"file://{config.base_dir / 'mlruns'}")
        
#         # Test database connection
#         test_id = metadata_db.log_dataset(
#             name="test_setup",
#             version="v1",
#             type_="test",
#             size=1,
#             file_path="/tmp/test",
#             metadata={"test": True}
#         )
        
#         logger.info(f"Environment setup completed successfully (test DB ID: {test_id})")
        
#         return {
#             "status": "success",
#             "timestamp": datetime.now().isoformat(),
#             "config_loaded": True,
#             "database_connected": True,
#             "mlflow_ready": True
#         }
        
#     except Exception as e:
#         logger.error(f"Environment setup failed: {str(e)}")
#         raise

# @task(name="data_ingestion", description="Ingest and preprocess training data")
# def data_ingestion_task(source_type: str = "huggingface",
#                        source_path: str = "wikitext",
#                        dataset_name: str = "training_data",
#                        max_samples: int = 1000) -> Dict[str, Any]:
#     """Data ingestion task"""
#     logger = get_run_logger()
#     logger.info(f"Starting data ingestion for {dataset_name}...")
    
#     try:
#         # Initialize ingestion pipeline
#         ingestion = DataIngestion(config.model_configs, metadata_db, config.data_dir)
        
#         # Run ingestion pipeline
#         saved_paths = ingestion.run_ingestion_pipeline(
#             source_type=source_type,
#             source_path=source_path,
#             dataset_name=dataset_name,
#             max_samples=max_samples
#         )
        
#         logger.info(f"Data ingestion completed. Files: {list(saved_paths.keys())}")
        
#         return {
#             "status": "success",
#             "dataset_name": dataset_name,
#             "saved_paths": {k: str(v) for k, v in saved_paths.items()},
#             "total_samples": max_samples,
#             "splits": list(saved_paths.keys())
#         }
        
#     except Exception as e:
#         logger.error(f"Data ingestion failed: {str(e)}")
#         raise

# @task(name="model_training", description="Fine-tune model with LoRA")
# def model_training_task(training_data_path: str,
#                        validation_data_path: Optional[str] = None,
#                        experiment_name: str = "prefect_training") -> Dict[str, Any]:
#     """Model training task"""
#     logger = get_run_logger()
#     logger.info(f"Starting model training for {experiment_name}...")
    
#     try:
#         # Initialize generator
#         generator = SyntheticTextGenerator(config.model_configs, metadata_db, config.models_dir)
        
#         # Load training data
#         with open(training_data_path, 'r', encoding='utf-8') as f:
#             train_data = json.load(f)
#         train_texts = train_data['texts']
        
#         # Load validation data if provided
#         val_texts = None
#         if validation_data_path and Path(validation_data_path).exists():
#             with open(validation_data_path, 'r', encoding='utf-8') as f:
#                 val_data = json.load(f)
#             val_texts = val_data['texts']
        
#         # Fine-tune model
#         model_path = generator.fine_tune_with_lora(
#             train_texts=train_texts,
#             val_texts=val_texts,
#             experiment_name=experiment_name
#         )
        
#         logger.info(f"Model training completed. Saved to: {model_path}")
        
#         return {
#             "status": "success",
#             "experiment_name": experiment_name,
#             "model_path": model_path,
#             "training_samples": len(train_texts),
#             "validation_samples": len(val_texts) if val_texts else 0
#         }
        
#     except Exception as e:
#         logger.error(f"Model training failed: {str(e)}")
#         raise

# @task(name="synthetic_generation", description="Generate synthetic dataset")
# def synthetic_generation_task(model_path: Optional[str] = None,
#                              num_samples: int = 500,
#                              dataset_name: str = "synthetic_batch") -> Dict[str, Any]:
#     """Synthetic data generation task"""
#     logger = get_run_logger()
#     logger.info(f"Starting synthetic generation for {dataset_name}...")
    
#     try:
#         # Initialize generator
#         generator = SyntheticTextGenerator(config.model_configs, metadata_db, config.models_dir)
        
#         # Load model
#         if model_path:
#             generator.load_finetuned_model(model_path)
#         else:
#             generator.load_base_model()
        
#         # Generate synthetic dataset
#         synthetic_texts = generator.generate_dataset(num_samples=num_samples)
        
#         # Save dataset
#         metadata = {
#             "prefect_flow": "synthetic_generation",
#             "model_used": model_path or "base_model",
#             "generation_timestamp": datetime.now().isoformat()
#         }
        
#         saved_path = generator.save_synthetic_dataset(
#             synthetic_texts,
#             dataset_name,
#             metadata=metadata
#         )
        
#         logger.info(f"Synthetic generation completed. Generated {len(synthetic_texts)} samples")
        
#         return {
#             "status": "success",
#             "dataset_name": dataset_name,
#             "saved_path": str(saved_path),
#             "generated_samples": len(synthetic_texts),
#             "model_used": model_path or "base_model"
#         }
        
#     except Exception as e:
#         logger.error(f"Synthetic generation failed: {str(e)}")
#         raise

# @task(name="dataset_validation", description="Validate synthetic dataset quality")
# def dataset_validation_task(synthetic_dataset_path: str,
#                           real_dataset_path: Optional[str] = None,
#                           dataset_name: str = "validation") -> Dict[str, Any]:
#     """Dataset validation task"""
#     logger = get_run_logger()
#     logger.info(f"Starting validation for {dataset_name}...")
    
#     try:
#         # Initialize validation pipeline
#         validation_pipeline = ValidationPipeline(config.model_configs, metadata_db)
        
#         # Convert string paths to Path objects
#         synthetic_path = Path(synthetic_dataset_path)
#         real_path = Path(real_dataset_path) if real_dataset_path else None
        
#         # Create output directory for reports
#         output_dir = config.base_dir / "reports" / "validation"
#         output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Run validation
#         results = validation_pipeline.validate_synthetic_dataset(
#             synthetic_dataset_path=synthetic_path,
#             real_dataset_path=real_path,
#             output_dir=output_dir
#         )
        
#         logger.info(f"Validation completed. Overall passed: {results['overall_passed']}")
        
#         return {
#             "status": "success",
#             "dataset_name": dataset_name,
#             "overall_passed": results['overall_passed'],
#             "failed_checks": results['failed_checks'],
#             "total_samples": results['total_samples'],
#             "validation_details": results['validations'],
#             "report_dir": str(output_dir)
#         }
        
#     except Exception as e:
#         logger.error(f"Dataset validation failed: {str(e)}")
#         raise

# @task(name="quality_monitoring", description="Monitor pipeline quality metrics")
# def quality_monitoring_task() -> Dict[str, Any]:
#     """Monitor overall pipeline quality and performance"""
#     logger = get_run_logger()
#     logger.info("Running quality monitoring...")
    
#     try:
#         # Query database for recent metrics
#         import sqlite3
        
#         with sqlite3.connect(metadata_db.db_path) as conn:
#             # Get recent datasets
#             cursor = conn.cursor()
#             cursor.execute("""
#                 SELECT type, COUNT(*) as count, AVG(size) as avg_size
#                 FROM datasets 
#                 WHERE created_at > datetime('now', '-7 days')
#                 GROUP BY type
#             """)
#             recent_datasets = {row[0]: {"count": row[1], "avg_size": row[2]} 
#                              for row in cursor.fetchall()}
            
#             # Get validation success rates
#             cursor.execute("""
#                 SELECT validation_type, 
#                        SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed,
#                        COUNT(*) as total
#                 FROM validations 
#                 WHERE created_at > datetime('now', '-7 days')
#                 GROUP BY validation_type
#             """)
#             validation_rates = {}
#             for row in cursor.fetchall():
#                 validation_rates[row[0]] = {
#                     "success_rate": row[1] / row[2] if row[2] > 0 else 0,
#                     "total_validations": row[2]
#                 }
        
#         # Calculate overall health score
#         health_score = 1.0
#         if 'quality' in validation_rates:
#             health_score *= validation_rates['quality']['success_rate']
#         if 'privacy' in validation_rates:
#             health_score *= validation_rates['privacy']['success_rate']
        
#         # Create monitoring report
#         monitoring_report = {
#             "timestamp": datetime.now().isoformat(),
#             "recent_datasets": recent_datasets,
#             "validation_rates": validation_rates,
#             "health_score": health_score,
#             "status": "healthy" if health_score > 0.8 else "warning" if health_score > 0.5 else "critical"
#         }
        
#         logger.info(f"Quality monitoring completed. Health score: {health_score:.2f}")
        
#         return monitoring_report
        
#     except Exception as e:
#         logger.error(f"Quality monitoring failed: {str(e)}")
#         raise

# @task(name="cleanup_old_files", description="Clean up old files and artifacts")
# def cleanup_task(days_to_keep: int = 30) -> Dict[str, Any]:
#     """Clean up old files and artifacts"""
#     logger = get_run_logger()
#     logger.info(f"Cleaning up files older than {days_to_keep} days...")
    
#     try:
#         cleanup_stats = {
#             "deleted_files": 0,
#             "freed_space_mb": 0,
#             "errors": []
#         }
        
#         cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
#         # Clean up temporary files
#         temp_dirs = [
#             config.data_dir / "temp",
#             config.models_dir / "temp",
#             config.logs_dir / "temp"
#         ]
        
#         for temp_dir in temp_dirs:
#             if temp_dir.exists():
#                 for file_path in temp_dir.rglob("*"):
#                     if file_path.is_file():
#                         try:
#                             file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
#                             if file_time < cutoff_date:
#                                 file_size = file_path.stat().st_size
#                                 file_path.unlink()
#                                 cleanup_stats["deleted_files"] += 1
#                                 cleanup_stats["freed_space_mb"] += file_size / (1024 * 1024)
#                         except Exception as e:
#                             cleanup_stats["errors"].append(str(e))
        
#         logger.info(f"Cleanup completed. Deleted {cleanup_stats['deleted_files']} files, "
#                    f"freed {cleanup_stats['freed_space_mb']:.2f} MB")
        
#         return {
#             "status": "success",
#             "cleanup_stats": cleanup_stats,
#             "days_kept": days_to_keep
#         }
        
#     except Exception as e:
#         logger.error(f"Cleanup failed: {str(e)}")
#         raise

# # Prefect flows
# @flow(name="End-to-End Pipeline", 
#       description="Complete synthetic data generation pipeline",
#       task_runner=SequentialTaskRunner())
# def end_to_end_pipeline(source_type: str = "huggingface",
#                        source_path: str = "wikitext", 
#                        dataset_name: str = "e2e_pipeline",
#                        max_samples: int = 1000,
#                        num_synthetic: int = 500,
#                        run_training: bool = True) -> Dict[str, Any]:
#     """Complete end-to-end pipeline flow"""
#     logger = get_run_logger()
#     logger.info("Starting end-to-end synthetic data pipeline...")
    
#     # 1. Setup environment
#     env_result = setup_environment_task()
    
#     # 2. Data ingestion
#     ingestion_result = data_ingestion_task(
#         source_type=source_type,
#         source_path=source_path,
#         dataset_name=dataset_name,
#         max_samples=max_samples
#     )
    
#     # 3. Model training (optional)
#     model_path = None
#     if run_training:
#         training_result = model_training_task(
#             training_data_path=ingestion_result["saved_paths"]["train"],
#             validation_data_path=ingestion_result["saved_paths"].get("validation"),
#             experiment_name=f"{dataset_name}_training"
#         )
#         model_path = training_result["model_path"]
    
#     # 4. Synthetic generation
#     generation_result = synthetic_generation_task(
#         model_path=model_path,
#         num_samples=num_synthetic,
#         dataset_name=f"{dataset_name}_synthetic"
#     )
    
#     # 5. Validation
#     validation_result = dataset_validation_task(
#         synthetic_dataset_path=generation_result["saved_path"],
#         real_dataset_path=ingestion_result["saved_paths"]["test"],
#         dataset_name=f"{dataset_name}_validation"
#     )
    
#     # 6. Quality monitoring
#     monitoring_result = quality_monitoring_task()
    
#     # Compile final results
#     pipeline_result = {
#         "pipeline_status": "success",
#         "timestamp": datetime.now().isoformat(),
#         "dataset_name": dataset_name,
#         "environment": env_result,
#         "ingestion": ingestion_result,
#         "generation": generation_result,
#         "validation": validation_result,
#         "monitoring": monitoring_result
#     }
    
#     if run_training:
#         pipeline_result["training"] = training_result
    
#     logger.info(f"End-to-end pipeline completed successfully for {dataset_name}")
#     return pipeline_result

# @flow(name="Scheduled Generation", 
#       description="Scheduled synthetic data generation",
#       task_runner=ConcurrentTaskRunner())
# def scheduled_generation_flow(num_samples: int = 100,
#                             dataset_prefix: str = "daily") -> Dict[str, Any]:
#     """Scheduled flow for regular synthetic data generation"""
#     logger = get_run_logger()
#     logger.info(f"Starting scheduled generation of {num_samples} samples...")
    
#     # Setup
#     env_result = setup_environment_task()
    
#     # Generate with current timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M")
#     dataset_name = f"{dataset_prefix}_{timestamp}"
    
#     # Generate synthetic data
#     generation_result = synthetic_generation_task(
#         model_path=None,  # Use base model
#         num_samples=num_samples,
#         dataset_name=dataset_name
#     )
    
#     # Quick validation
#     validation_result = dataset_validation_task(
#         synthetic_dataset_path=generation_result["saved_path"],
#         dataset_name=f"{dataset_name}_validation"
#     )
    
#     # Monitor quality
#     monitoring_result = quality_monitoring_task()
    
#     return {
#         "status": "success",
#         "timestamp": datetime.now().isoformat(),
#         "generation": generation_result,
#         "validation": validation_result,
#         "monitoring": monitoring_result
#     }

# @flow(name="Model Training Pipeline",
#       description="Dedicated model training and evaluation pipeline",
#       task_runner=SequentialTaskRunner())
# def model_training_pipeline(training_data_path: str,
#                           validation_data_path: Optional[str] = None,
#                           experiment_name: str = "training_pipeline") -> Dict[str, Any]:
#     """Dedicated model training pipeline"""
#     logger = get_run_logger()
#     logger.info(f"Starting model training pipeline: {experiment_name}")
    
#     # Setup
#     env_result = setup_environment_task()
    
#     # Train model
#     training_result = model_training_task(
#         training_data_path=training_data_path,
#         validation_data_path=validation_data_path,
#         experiment_name=experiment_name
#     )
    
#     # Test generation with trained model
#     test_generation_result = synthetic_generation_task(
#         model_path=training_result["model_path"],
#         num_samples=50,  # Small test set
#         dataset_name=f"{experiment_name}_test"
#     )
    
#     # Validate test generation
#     test_validation_result = dataset_validation_task(
#         synthetic_dataset_path=test_generation_result["saved_path"],
#         dataset_name=f"{experiment_name}_test_validation"
#     )
    
#     return {
#         "status": "success",
#         "timestamp": datetime.now().isoformat(),
#         "experiment_name": experiment_name,
#         "environment": env_result,
#         "training": training_result,
#         "test_generation": test_generation_result,
#         "test_validation": test_validation_result
#     }

# @flow(name="Monitoring and Maintenance",
#       description="System monitoring and maintenance flow",
#       task_runner=ConcurrentTaskRunner())
# def monitoring_maintenance_flow(cleanup_days: int = 30) -> Dict[str, Any]:
#     """System monitoring and maintenance flow"""
#     logger = get_run_logger()
#     logger.info("Starting monitoring and maintenance flow...")
    
#     # Quality monitoring
#     monitoring_result = quality_monitoring_task()
    
#     # Cleanup old files
#     cleanup_result = cleanup_task(days_to_keep=cleanup_days)
    
#     return {
#         "status": "success",
#         "timestamp": datetime.now().isoformat(),
#         "monitoring": monitoring_result,
#         "cleanup": cleanup_result
#     }

# # Deployment configurations for scheduled flows
# def create_scheduled_deployments():
#     """Create Prefect deployments with schedules"""
    
#     # Daily generation at 2 AM
#     daily_generation_deployment = scheduled_generation_flow.to_deployment(
#         name="daily-synthetic-generation",
#         description="Generate synthetic data daily",
#         schedule=CronSchedule(cron="0 2 * * *", timezone="UTC"),
#         parameters={"num_samples": 200, "dataset_prefix": "daily"}
#     )
    
#     # Weekly maintenance on Sundays at 3 AM
#     weekly_maintenance_deployment = monitoring_maintenance_flow.to_deployment(
#         name="weekly-maintenance",
#         description="Weekly system maintenance",
#         schedule=CronSchedule(cron="0 3 * * 0", timezone="UTC"),
#         parameters={"cleanup_days": 30}
#     )
    
#     return [daily_generation_deployment, weekly_maintenance_deployment]

# # Main execution for testing
# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Run Prefect pipeline flows")
#     parser.add_argument("--flow", 
#                        choices=["e2e", "scheduled", "training", "monitoring"],
#                        default="e2e",
#                        help="Flow to run")
#     parser.add_argument("--dataset-name", default="test_pipeline",
#                        help="Dataset name")
#     parser.add_argument("--max-samples", type=int, default=500,
#                        help="Maximum samples to process")
#     parser.add_argument("--num-synthetic", type=int, default=100,
#                        help="Number of synthetic samples to generate")
#     parser.add_argument("--no-training", action="store_true",
#                        help="Skip model training in e2e pipeline")
    
#     args = parser.parse_args()
    
#     # Run selected flow
#     if args.flow == "e2e":
#         result = end_to_end_pipeline(
#             dataset_name=args.dataset_name,
#             max_samples=args.max_samples,
#             num_synthetic=args.num_synthetic,
#             run_training=not args.no_training
#         )
#         print(f"✅ End-to-end pipeline completed: {result['pipeline_status']}")
        
#     elif args.flow == "scheduled":
#         result = scheduled_generation_flow(
#             num_samples=args.num_synthetic,
#             dataset_prefix=args.dataset_name
#         )
#         print(f"✅ Scheduled generation completed: {result['status']}")
        
#     elif args.flow == "training":
#         # This would need training data path - simplified for demo
#         print("Training pipeline requires training data path. Use the API or modify the script.")
        
#     elif args.flow == "monitoring":
#         result = monitoring_maintenance_flow()
#         print(f"✅ Monitoring and maintenance completed: {result['status']}")
#         print(f"Health score: {result['monitoring']['health_score']:.2f}")



# pipelines/prefect/flows.py
"""
Prefect flows for orchestrating the synthetic data generation pipeline.
End-to-end (ingest -> optional LoRA finetune -> generate -> validate), scheduled generation,
standalone training, and monitoring/maintenance.

Key upgrades vs. previous version:
- Full parameterization for all modes (ingestion, training, generation, validation).
- Training switch: LoRA finetune OR pure prompting (zero/one/few-shot).
- **Dataset-aware prompt builder** (AG News / IMDB): label-conditioned prompts per class.
- Few-shot examples: file-based (JSON/JSONL/CSV) or generic in-context scaffolding across labels.
- Generation hyperparameters (temperature/top_k/top_p/repetition_penalty/max_length/num_return_sequences).
- LoRA hyperparameters override (r, alpha, dropout, target modules).
- Stronger error handling, seeding, MLflow URI control, consistent return payloads.
"""

from __future__ import annotations

import os
import csv
import json
import math
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner, ConcurrentTaskRunner
from prefect.server.schemas.schedules import CronSchedule
import mlflow

# ---------- Project paths & imports ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # pipelines/prefect -> pipelines -> repo root
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Adjust these imports to your package layout if needed
from src.pipeline_setup import Config, MetadataDB, setup_logging
from src.ingestion import DataIngestion
from src.generator import SyntheticTextGenerator
from src.validate import ValidationPipeline

# ---------- Global singletons ----------
config = Config()
metadata_db = MetadataDB(config.data_dir / "pipeline.db")


# ---------- Dataset labels & prompt templates ----------
DATASET_LABELS: Dict[str, Dict[int, str]] = {
    "ag_news": {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech",
    },
    "imdb": {
        0: "negative",
        1: "positive",
    },
}

AG_ZERO_TMPL = (
    "You are a news editor. Write a concise news summary for the category: {label_name}.\n"
    "- Keep it factual and varied.\n"
    "- Include plausible named entities.\n"
    "Length: 40-80 words.\n"
    "\n### Task:\nWrite one {label_name} news summary."
)

IMDB_ZERO_TMPL = (
    "You are an IMDB movie reviewer. Write a concise {label_name} review.\n"
    "- Keep it natural and varied.\n"
    "- Avoid explicit content.\n"
    "Length: 60-120 words.\n"
    "\n### Task:\nWrite one {label_name} movie review."
)


def _set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _maybe_set_mlflow_tracking(uri: Optional[str]) -> None:
    if uri:
        mlflow.set_tracking_uri(uri)
    elif not mlflow.get_tracking_uri():
        # Default to local file store
        mlflow.set_tracking_uri(f"file://{config.base_dir / 'mlruns'}")
    os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if isinstance(data, list):
                items = data
        else:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _load_examples(examples_path: Optional[str]) -> List[Dict[str, Any]]:
    """Load in-context examples with flexible schema. Accepts JSON/JSONL/CSV.
    Tries keys: ('prompt','completion') or ('instruction','input','output')."""
    if not examples_path:
        return []
    p = Path(examples_path)
    if not p.exists():
        raise FileNotFoundError(f"Examples file not found: {examples_path}")
    if p.suffix.lower() in {".json", ".jsonl"}:
        return _read_json_or_jsonl(p)
    if p.suffix.lower() == ".csv":
        return _read_csv(p)
    raise ValueError(f"Unsupported examples format: {p.suffix}")


def _example_to_text(ex: Dict[str, Any]) -> str:
    """Normalize an example row into a single in-context block of text."""
    if "prompt" in ex and "completion" in ex:
        return f"### Prompt:\n{ex['prompt'].strip()}\n\n### Response:\n{ex['completion'].strip()}\n"
    if "instruction" in ex and "output" in ex:
        instr = ex["instruction"].strip()
        inp = ex.get("input", "")
        if inp:
            return f"### Instruction:\n{instr}\n\n### Input:\n{inp.strip()}\n\n### Response:\n{ex['output'].strip()}\n"
        return f"### Instruction:\n{instr}\n\n### Response:\n{ex['output'].strip()}\n"
    # Fallback: join all fields
    parts = [f"{k}: {v}" for k, v in ex.items()]
    return "\n".join(parts)


def _build_in_context_prompt(
    base_prompt: str,
    examples: List[Dict[str, Any]],
    shots: int,
    example_strategy: str = "head"  # or "random"
) -> str:
    if not examples or shots <= 0:
        return base_prompt
    if example_strategy == "random":
        exs = random.sample(examples, k=min(shots, len(examples)))
    else:
        exs = examples[:shots]
    blocks = "\n".join(_example_to_text(e) for e in exs)
    return f"{blocks}\n\n### Task:\n{base_prompt.strip()}\n"


def _load_prompts_file(prompts_file: Optional[str]) -> List[str]:
    if not prompts_file:
        return []
    p = Path(prompts_file)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    prompts: List[str] = []
    if p.suffix.lower() in {".json", ".jsonl"}:
        rows = _read_json_or_jsonl(p)
        for r in rows:
            if isinstance(r, str):
                prompts.append(r)
            elif "prompt" in r:
                prompts.append(str(r["prompt"]))
            elif "instruction" in r:
                base = str(r["instruction"])
                if r.get("input"):
                    base += f"\n\nInput: {r['input']}"
                prompts.append(base)
    else:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
    return prompts


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _label_conditioned_base_prompts(dataset_kind: str) -> List[str]:
    """
    Build one base prompt per label (balanced generation).
    """
    dk = dataset_kind.lower()
    if dk not in DATASET_LABELS:
        raise ValueError(f"Unsupported dataset_kind: {dataset_kind}")
    labels = DATASET_LABELS[dk]
    prompts: List[str] = []
    for _, name in labels.items():
        if dk == "ag_news":
            prompts.append(AG_ZERO_TMPL.format(label_name=name))
        else:
            prompts.append(IMDB_ZERO_TMPL.format(label_name=name))
    return prompts


# ---------- Prefect tasks ----------
@task(name="setup_environment", description="Setup pipeline environment and check dependencies")
def setup_environment_task(mlflow_uri: Optional[str] = None, log_level: str = "INFO") -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Setting up pipeline environment...")
    try:
        setup_logging(config.logs_dir, getattr(logging, log_level.upper(), logging.INFO))
        for d in [config.data_dir, config.models_dir, config.logs_dir, config.reports_dir]:
            _ensure_dir(d)
        # _maybe_set_mlflow_tracking(mlflow_uri)

        # Smoke test DB
        test_id = metadata_db.log_dataset(
            name="env_setup_smoke",
            version="v0",
            type_="smoke",
            size=1,
            file_path=str(config.data_dir / "env_setup_smoke"),
            metadata={"ok": True}
        )
        logger.info(f"Environment OK. DB id: {test_id}")
        return {"status": "success", "db_id": test_id, "mlflow_uri": mlflow.get_tracking_uri()}
    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        raise


@task(name="data_ingestion", description="Ingest and preprocess data")
def data_ingestion_task(
    source_type: str = "huggingface",
    source_path: str = "ag_news",  # default to AG News
    dataset_name: str = "dataset",
    version: str = "v1",
    max_samples: int = 1000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info(f"Ingesting data: {dataset_name} ({source_type}:{source_path})")
    try:
        ingestion = DataIngestion(config.model_configs, metadata_db, config.data_dir)
        saved_paths = ingestion.run_ingestion_pipeline(
            source_type=source_type,
            source_path=source_path,  # e.g., "ag_news" or "imdb"
            dataset_name=dataset_name,
            version=version,
            max_samples=max_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "version": version,
            "saved_paths": {k: str(v) for k, v in saved_paths.items()},
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


@task(name="model_training", description="Optionally fine-tune model with LoRA")
def model_training_task(
    training_method: str = "lora",  # "lora" or "none"
    training_data_path: Optional[str] = None,
    validation_data_path: Optional[str] = None,
    base_model_name: Optional[str] = None,  # override default
    experiment_name: str = "prefect_training",
    seed: Optional[int] = 42,
    # LoRA overrides
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    lora_target_modules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info(f"Training method: {training_method}")
    try:
        if training_method != "lora":
            logger.info("Skipping training (prompt-only).")
            return {"status": "skipped", "training_method": training_method, "model_path": None}

        if not training_data_path:
            raise ValueError("training_data_path is required when training_method='lora'")

        _set_global_seed(seed)

        # Load data
        with open(training_data_path, "r", encoding="utf-8") as f:
            train_texts = json.load(f)["texts"]
        val_texts = None
        if validation_data_path and Path(validation_data_path).exists():
            with open(validation_data_path, "r", encoding="utf-8") as f:
                val_texts = json.load(f)["texts"]

        generator = SyntheticTextGenerator(config.model_configs, metadata_db, config.models_dir)

        # Base model override if provided
        if base_model_name:
            generator.load_base_model(base_model_name)

        # Optional LoRA overrides
        custom_lora = {}
        if lora_r is not None: custom_lora["r"] = lora_r
        if lora_alpha is not None: custom_lora["lora_alpha"] = lora_alpha
        if lora_dropout is not None: custom_lora["lora_dropout"] = lora_dropout
        if lora_target_modules is not None: custom_lora["target_modules"] = lora_target_modules
        if custom_lora and hasattr(generator, "setup_peft_config"):
            generator.setup_peft_config(custom_lora)

        model_path = generator.fine_tune_with_lora(
            train_texts=train_texts,
            val_texts=val_texts,
            experiment_name=experiment_name,
        )
        logger.info(f"LoRA model saved to: {model_path}")
        return {
            "status": "success",
            "training_method": "lora",
            "model_path": model_path,
            "training_samples": len(train_texts),
            "validation_samples": len(val_texts) if val_texts else 0,
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


@task(name="prepare_prompts", description="Build prompts for zero/one/few-shot generation (label-conditioned)")
def prepare_prompts_task(
    dataset_kind: str = "ag_news",        # "ag_news" | "imdb"
    prompt_mode: str = "zero",            # "zero" | "one" | "few"
    base_prompt: Optional[str] = None,    # optional override (applied per label)
    prompts_file: Optional[str] = None,   # if provided, used directly (no label balancing)
    examples_path: Optional[str] = None,  # optional: JSON/JSONL/CSV for in-context
    shots: int = 0,
    example_strategy: str = "head",       # or "random"
) -> Dict[str, Any]:
    """
    Returns a list of prompts. By default, creates one prompt per label (balanced generation),
    with dataset-specific templates. Few-shot wraps each base prompt with k examples.
    """
    logger = get_run_logger()
    try:
        # If a prompts file is provided, use it directly (caller takes responsibility for balancing)
        if prompts_file:
            prompts = _load_prompts_file(prompts_file)
            logger.info(f"Prepared {len(prompts)} prompt(s) from file.")
            return {"status": "success", "prompts": prompts, "labels": None}

        # Build dataset-specific, label-conditioned prompts
        base_prompts = _label_conditioned_base_prompts(dataset_kind)
        if base_prompt:
            # If the user supplies a generic base_prompt, inject label names into it
            labels = list(DATASET_LABELS[dataset_kind].values())
            base_prompts = [base_prompt.format(label_name=ln) for ln in labels]

        # Few-shot / one-shot handling via examples
        examples = _load_examples(examples_path) if examples_path else []
        k = 0 if prompt_mode == "zero" else 1 if prompt_mode == "one" else max(2, shots)
        prompts = [_build_in_context_prompt(p, examples, k, example_strategy) for p in base_prompts]
        logger.info(f"Prepared {len(prompts)} label-conditioned prompt(s) for {dataset_kind} (mode={prompt_mode}, k={k}).")
        return {"status": "success", "prompts": prompts, "labels": DATASET_LABELS[dataset_kind]}
    except Exception as e:
        logger.error(f"Prompt preparation failed: {e}")
        raise

@task(name="synthetic_generation", description="Generate synthetic dataset (balanced across labels if prompts are per-label)")
def synthetic_generation_task(
    model_path: Optional[str] = None,
    base_model_name: Optional[str] = None,
    prompts: Optional[List[str]] = None,
    num_samples: Optional[int] = None,            # if None, derive from synth_per_label * len(prompts)
    synth_per_label: int = 500,                   # balanced generation per prompt (label)
    dataset_name: str = "synthetic_batch",
    # decoding params (pass-through)
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    max_length: int = 256,
    num_return_sequences: int = 1,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    logger = get_run_logger()
    try:
        _set_global_seed(seed)
        gen = SyntheticTextGenerator(config.model_configs, metadata_db, config.models_dir)

        # Load model (finetuned adapter or base)
        if model_path:
            gen.load_finetuned_model(model_path)
            model_used = model_path
        else:
            gen.load_base_model(base_model_name) if base_model_name else gen.load_base_model()
            # try to read what actually got loaded from the generator
            model_used = getattr(gen, "current_model_name", None) or base_model_name or "base_model"

        if not prompts:
            raise ValueError("No prompts provided to generation task.")

        # Decide total samples and how many per prompt (balanced)
        total_needed = num_samples if num_samples is not None else synth_per_label * len(prompts)
        per_prompt = max(1, math.ceil(total_needed / max(1, len(prompts))))

        generated: List[str] = []
        # Helper to call generate_text (preferred) or generate_dataset with full kwargs
        def _gen_with_kwargs(prompt: str, n: int) -> List[str]:
            if hasattr(gen, "generate_text"):
                return gen.generate_text(
                    prompt=prompt,
                    num_sequences=n * max(1, num_return_sequences),
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
            elif hasattr(gen, "generate_dataset"):
                # forward decoding knobs to the dataset generator if it supports **kwargs
                return gen.generate_dataset(
                    prompts=[prompt],
                    num_samples=n * max(1, num_return_sequences),
                    samples_per_prompt=1,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=1,
                )
            else:
                raise RuntimeError("SyntheticTextGenerator must implement generate_text or generate_dataset.")

        # First pass: balanced across prompts
        for p in prompts:
            remaining = total_needed - len(generated)
            if remaining <= 0:
                break
            n = min(per_prompt, remaining)
            seqs = _gen_with_kwargs(p, n)
            generated.extend(seqs)

        # If still short (round-robin prompts)
        i = 0
        while len(generated) < total_needed:
            p = prompts[i % len(prompts)]
            remain = total_needed - len(generated)
            seqs = _gen_with_kwargs(p, min(per_prompt, remain))
            generated.extend(seqs)
            i += 1

        # Trim to exactly total_needed
        generated = generated[:total_needed]

        # Metadata (what was actually used)
        meta = {
            "prefect_flow": "synthetic_generation",
            "model_used": model_used,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_length": max_length,
            "num_return_sequences": num_return_sequences,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        }
        saved_path = gen.save_synthetic_dataset(generated, dataset_name, metadata=meta)
        logger.info(f"Generated {len(generated)} samples -> {saved_path}")
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "saved_path": str(saved_path),
            "generated_samples": len(generated),
            "model_used": model_used,
        }
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise



@task(name="dataset_validation", description="Validate synthetic dataset quality")
def dataset_validation_task(
    synthetic_dataset_path: str,
    real_dataset_path: Optional[str] = None,
    dataset_name: str = "validation",
    output_subdir: str = "validation",
) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info(f"Validating dataset: {dataset_name}")
    try:
        validation_pipeline = ValidationPipeline(config.model_configs, metadata_db)
        out_dir = config.reports_dir / output_subdir
        _ensure_dir(out_dir)
        results = validation_pipeline.validate_synthetic_dataset(
            synthetic_dataset_path=Path(synthetic_dataset_path),
            real_dataset_path=Path(real_dataset_path) if real_dataset_path else None,
            output_dir=out_dir,
        )
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "overall_passed": results.get("overall_passed", False),
            "failed_checks": results.get("failed_checks", []),
            "total_samples": results.get("total_samples"),
            "validation_details": results.get("validations"),
            "report_dir": str(out_dir),
        }
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


@task(name="quality_monitoring", description="Monitor pipeline quality metrics")
def quality_monitoring_task() -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Running quality monitoring...")
    try:
        import sqlite3
        with sqlite3.connect(metadata_db.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT type, COUNT(*) as count, AVG(size) as avg_size
                FROM datasets 
                WHERE created_at > datetime('now', '-7 days')
                GROUP BY type
            """)
            recent = {row[0]: {"count": row[1], "avg_size": row[2]} for row in cur.fetchall()}
            cur.execute("""
                SELECT validation_type, 
                       SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed,
                       COUNT(*) as total
                FROM validations 
                WHERE created_at > datetime('now', '-7 days')
                GROUP BY validation_type
            """)
            rates = {}
            for row in cur.fetchall():
                rates[row[0]] = {
                    "success_rate": (row[1] / row[2]) if row[2] else 0.0,
                    "total": row[2],
                }
        health = 1.0
        for k, r in rates.items():
            health *= (0.5 + 0.5 * r["success_rate"])  # smoother multiplicative score
        status = "healthy" if health > 0.8 else "warning" if health > 0.6 else "critical"
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "recent_datasets": recent,
            "validation_rates": rates,
            "health_score": health,
            "overall_status": status,
        }
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise


@task(name="cleanup_old_files", description="Clean up old files and artifacts")
def cleanup_task(days_to_keep: int = 30) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info(f"Cleaning files older than {days_to_keep} days...")
    try:
        stats = {"deleted_files": 0, "freed_space_mb": 0.0, "errors": []}
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        for root in [config.data_dir, config.models_dir, config.logs_dir, config.reports_dir]:
            temp = root / "temp"
            if not temp.exists():
                continue
            for fp in temp.rglob("*"):
                if not fp.is_file():
                    continue
                try:
                    mtime = datetime.fromtimestamp(fp.stat().st_mtime)
                    if mtime < cutoff:
                        size_mb = fp.stat().st_size / (1024 * 1024)
                        fp.unlink()
                        stats["deleted_files"] += 1
                        stats["freed_space_mb"] += size_mb
                except Exception as e:
                    stats["errors"].append(str(e))
        return {"status": "success", "cleanup_stats": stats, "days_kept": days_to_keep}
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise


# ---------- Prefect flows ----------
@flow(
    name="End-to-End Pipeline",
    description="Ingest -> (optional LoRA finetune) -> generate -> validate",
    task_runner=SequentialTaskRunner(),
)
def end_to_end_pipeline(
    # environment
    mlflow_uri: Optional[str] = None,
    log_level: str = "INFO",
    seed: Optional[int] = 42,
    # ingestion
    dataset_kind: str = "ag_news",      # NEW: "ag_news" | "imdb"
    source_type: str = "huggingface",
    source_path: Optional[str] = None,  # if None, auto from dataset_kind
    dataset_name: str = "e2e_pipeline",
    version: str = "v1",
    max_samples: int = 1000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    # training
    training_method: str = "none",  # "lora" or "none"
    base_model_name: Optional[str] = None,
    experiment_name: str = "e2e_training",
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    lora_target_modules: Optional[List[str]] = None,
    # prompts (dataset-aware)
    prompt_mode: str = "zero",  # "zero" | "one" | "few"
    base_prompt: Optional[str] = None,   # override; can include {label_name}
    prompts_file: Optional[str] = None,
    examples_path: Optional[str] = None,
    shots: int = 0,
    example_strategy: str = "head",
    # generation
    synth_per_label: int = 500,         # NEW: balanced per label
    num_samples: Optional[int] = None,  # if provided, overrides synth_per_label*labels
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    max_length: int = 256,
    num_return_sequences: int = 1,
) -> Dict[str, Any]:
    logger = get_run_logger()
    _set_global_seed(seed)

    env = setup_environment_task(mlflow_uri=mlflow_uri, log_level=log_level)

    # pick correct HF dataset id
    src_path = source_path or dataset_kind
    ingest = data_ingestion_task(
        source_type=source_type,
        source_path=src_path,
        dataset_name=dataset_name,
        version=version,
        max_samples=max_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    training = model_training_task(
        training_method=training_method,
        training_data_path=ingest["saved_paths"].get("train"),
        validation_data_path=ingest["saved_paths"].get("validation"),
        base_model_name=base_model_name,
        experiment_name=experiment_name,
        seed=seed,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    model_path = training.get("model_path")

    prompts_pack = prepare_prompts_task(
        dataset_kind=dataset_kind,
        prompt_mode=prompt_mode,
        base_prompt=base_prompt,
        prompts_file=prompts_file,
        examples_path=examples_path,
        shots=shots,
        example_strategy=example_strategy,
    )

    gen = synthetic_generation_task(
        model_path=model_path,
        base_model_name=base_model_name,
        prompts=prompts_pack["prompts"],
        num_samples=num_samples,
        synth_per_label=synth_per_label,
        dataset_name=f"{dataset_name}_{version}_synthetic",
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        seed=seed,
    )

    val = dataset_validation_task(
        synthetic_dataset_path=gen["saved_path"],
        real_dataset_path=ingest["saved_paths"].get("test"),
        dataset_name=f"{dataset_name}_{version}_validation",
        output_subdir=f"{dataset_name}_{version}",
    )

    mon = quality_monitoring_task()

    result = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "environment": env,
        "ingestion": ingest,
        "training": training,
        "generation": gen,
        "validation": val,
        "monitoring": mon,
    }
    logger.info("End-to-end pipeline finished.")
    return result


@flow(
    name="Scheduled Generation",
    description="Regular synthetic data generation using base model or a provided finetuned model",
    task_runner=ConcurrentTaskRunner(),
)
def scheduled_generation_flow(
    mlflow_uri: Optional[str] = None,
    log_level: str = "INFO",
    seed: Optional[int] = 42,
    dataset_kind: str = "ag_news",        # NEW
    # generation config
    dataset_prefix: str = "daily",
    base_model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    prompt_mode: str = "zero",
    base_prompt: Optional[str] = None,
    prompts_file: Optional[str] = None,
    examples_path: Optional[str] = None,
    shots: int = 0,
    example_strategy: str = "head",
    synth_per_label: int = 200,           # NEW
    num_samples: Optional[int] = None,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    max_length: int = 256,
    num_return_sequences: int = 1,
) -> Dict[str, Any]:
    get_run_logger().info("Scheduled generation start")
    _set_global_seed(seed)
    env = setup_environment_task(mlflow_uri=mlflow_uri, log_level=log_level)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dataset_name = f"{dataset_prefix}_{timestamp}"

    prompts_pack = prepare_prompts_task(
        dataset_kind=dataset_kind,
        prompt_mode=prompt_mode,
        base_prompt=base_prompt,
        prompts_file=prompts_file,
        examples_path=examples_path,
        shots=shots,
        example_strategy=example_strategy,
    )

    gen = synthetic_generation_task(
        model_path=model_path,
        base_model_name=base_model_name,
        prompts=prompts_pack["prompts"],
        num_samples=num_samples,
        synth_per_label=synth_per_label,
        dataset_name=dataset_name,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        seed=seed,
    )

    val = dataset_validation_task(
        synthetic_dataset_path=gen["saved_path"],
        dataset_name=f"{dataset_name}_validation",
        output_subdir=f"{dataset_name}",
    )

    mon = quality_monitoring_task()

    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "environment": env,
        "generation": gen,
        "validation": val,
        "monitoring": mon,
    }


@flow(
    name="Model Training Pipeline",
    description="Standalone training (LoRA) + quick smoke generation/validation",
    task_runner=SequentialTaskRunner(),
)
def model_training_pipeline(
    mlflow_uri: Optional[str] = None,
    log_level: str = "INFO",
    seed: Optional[int] = 42,
    # data
    training_data_path: str = "",
    validation_data_path: Optional[str] = None,
    # training
    base_model_name: Optional[str] = None,
    experiment_name: str = "training_pipeline",
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    lora_target_modules: Optional[List[str]] = None,
    # small test generation
    base_prompt: str = "Write a short paragraph on the topic.",
    num_samples: int = 20,
    max_length: int = 128,
) -> Dict[str, Any]:
    _set_global_seed(seed)
    env = setup_environment_task(mlflow_uri=mlflow_uri, log_level=log_level)
    training = model_training_task(
        training_method="lora",
        training_data_path=training_data_path,
        validation_data_path=validation_data_path,
        base_model_name=base_model_name,
        experiment_name=experiment_name,
        seed=seed,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    gen = synthetic_generation_task(
        model_path=training["model_path"],
        prompts=[base_prompt],
        num_samples=num_samples,
        synth_per_label=max(1, num_samples),  # trivial balance for single prompt
        dataset_name=f"{experiment_name}_test",
        max_length=max_length,
        seed=seed,
    )
    val = dataset_validation_task(
        synthetic_dataset_path=gen["saved_path"],
        dataset_name=f"{experiment_name}_test_validation",
        output_subdir=f"{experiment_name}",
    )
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "environment": env,
        "training": training,
        "test_generation": gen,
        "test_validation": val,
    }


@flow(
    name="Monitoring and Maintenance",
    description="Quality monitoring and cleanup",
    task_runner=ConcurrentTaskRunner(),
)
def monitoring_maintenance_flow(cleanup_days: int = 30) -> Dict[str, Any]:
    mon = quality_monitoring_task()
    cleanup = cleanup_task(days_to_keep=cleanup_days)
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "monitoring": mon,
        "cleanup": cleanup,
    }


# ---------- Deployments ----------
def create_scheduled_deployments():
    daily_generation_deployment = scheduled_generation_flow.to_deployment(
        name="daily-synthetic-generation",
        description="Generate synthetic data daily",
        schedule=CronSchedule(cron="0 2 * * *", timezone="UTC"),
        parameters={"num_samples": None, "synth_per_label": 200, "dataset_prefix": "daily"},
    )
    weekly_maintenance_deployment = monitoring_maintenance_flow.to_deployment(
        name="weekly-maintenance",
        description="Weekly system maintenance",
        schedule=CronSchedule(cron="0 3 * * 0", timezone="UTC"),
        parameters={"cleanup_days": 30},
    )
    return [daily_generation_deployment, weekly_maintenance_deployment]


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Prefect pipeline flows")

    sub = parser.add_subparsers(dest="flow", required=True)

    # ---- e2e ----
    e2e = sub.add_parser("e2e", help="End-to-end: ingest -> (LoRA) -> generate -> validate")
    e2e.add_argument("--mlflow-uri")
    e2e.add_argument("--log-level", default="INFO")
    e2e.add_argument("--seed", type=int, default=42)

    e2e.add_argument("--dataset-kind", choices=["ag_news","imdb"], default="ag_news")
    e2e.add_argument("--source-type", default="huggingface")
    e2e.add_argument("--source-path")  # defaulted internally from dataset-kind
    e2e.add_argument("--dataset-name", default="e2e_pipeline")
    e2e.add_argument("--version", default="v1")
    e2e.add_argument("--max-samples", type=int, default=1000)
    e2e.add_argument("--train-ratio", type=float, default=0.8)
    e2e.add_argument("--val-ratio", type=float, default=0.1)
    e2e.add_argument("--test-ratio", type=float, default=0.1)

    e2e.add_argument("--training-method", choices=["lora", "none"], default="none")
    e2e.add_argument("--base-model-name")
    e2e.add_argument("--experiment-name", default="e2e_training")
    e2e.add_argument("--lora-r", type=int)
    e2e.add_argument("--lora-alpha", type=int)
    e2e.add_argument("--lora-dropout", type=float)
    e2e.add_argument("--lora-target-modules", nargs="*")

    e2e.add_argument("--prompt-mode", choices=["zero", "one", "few"], default="zero")
    e2e.add_argument("--base-prompt", help="Optional override; may include {label_name}")
    e2e.add_argument("--prompts-file")
    e2e.add_argument("--examples-path")
    e2e.add_argument("--shots", type=int, default=0)
    e2e.add_argument("--example-strategy", choices=["head", "random"], default="head")

    e2e.add_argument("--synth-per-label", type=int, default=500)
    e2e.add_argument("--num-samples", type=int)  # optional override
    e2e.add_argument("--temperature", type=float, default=0.8)
    e2e.add_argument("--top-k", type=int, default=50)
    e2e.add_argument("--top-p", type=float, default=0.95)
    e2e.add_argument("--repetition-penalty", type=float, default=1.0)
    e2e.add_argument("--max-length", type=int, default=256)
    e2e.add_argument("--num-return-sequences", type=int, default=1)

    # ---- scheduled ----
    sch = sub.add_parser("scheduled", help="Scheduled generation")
    sch.add_argument("--mlflow-uri")
    sch.add_argument("--log-level", default="INFO")
    sch.add_argument("--seed", type=int, default=42)
    sch.add_argument("--dataset-kind", choices=["ag_news","imdb"], default="ag_news")
    sch.add_argument("--dataset-prefix", default="daily")
    sch.add_argument("--base-model-name")
    sch.add_argument("--model-path")
    sch.add_argument("--prompt-mode", choices=["zero", "one", "few"], default="zero")
    sch.add_argument("--base-prompt")
    sch.add_argument("--prompts-file")
    sch.add_argument("--examples-path")
    sch.add_argument("--shots", type=int, default=0)
    sch.add_argument("--example-strategy", choices=["head", "random"], default="head")
    sch.add_argument("--synth-per-label", type=int, default=200)
    sch.add_argument("--num-samples", type=int)
    sch.add_argument("--temperature", type=float, default=0.8)
    sch.add_argument("--top-k", type=int, default=50)
    sch.add_argument("--top-p", type=float, default=0.95)
    sch.add_argument("--repetition-penalty", type=float, default=1.0)
    sch.add_argument("--max-length", type=int, default=256)
    sch.add_argument("--num-return-sequences", type=int, default=1)

    # ---- training ----
    tr = sub.add_parser("training", help="Standalone training (LoRA)")
    tr.add_argument("--mlflow-uri")
    tr.add_argument("--log-level", default="INFO")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--training-data-path", required=True)
    tr.add_argument("--validation-data-path")
    tr.add_argument("--base-model-name")
    tr.add_argument("--experiment-name", default="training_pipeline")
    tr.add_argument("--lora-r", type=int)
    tr.add_argument("--lora-alpha", type=int)
    tr.add_argument("--lora-dropout", type=float)
    tr.add_argument("--lora-target-modules", nargs="*")
    tr.add_argument("--base-prompt", default="Write a short paragraph on the topic.")
    tr.add_argument("--num-samples", type=int, default=20)
    tr.add_argument("--max-length", type=int, default=128)

    # ---- monitoring ----
    mon = sub.add_parser("monitoring", help="Monitoring and cleanup")
    mon.add_argument("--cleanup-days", type=int, default=30)

    args = parser.parse_args()

    if args.flow == "e2e":
        result = end_to_end_pipeline(
            mlflow_uri=args.mlflow_uri,
            log_level=args.log_level,
            seed=args.seed,
            dataset_kind=args.dataset_kind,
            source_type=args.source_type,
            source_path=args.source_path,
            dataset_name=args.dataset_name,
            version=args.version,
            max_samples=args.max_samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            training_method=args.training_method,
            base_model_name=args.base_model_name,
            experiment_name=args.experiment_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            prompt_mode=args.prompt_mode,
            base_prompt=args.base_prompt,
            prompts_file=args.prompts_file,
            examples_path=args.examples_path,
            shots=args.shots,
            example_strategy=args.example_strategy,
            synth_per_label=args.synth_per_label,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_length=args.max_length,
            num_return_sequences=args.num_return_sequences,
        )
        print(json.dumps({"ok": True, "flow": "e2e", "result": result["status"]}, indent=2))

    elif args.flow == "scheduled":
        result = scheduled_generation_flow(
            mlflow_uri=args.mlflow_uri,
            log_level=args.log_level,
            seed=args.seed,
            dataset_kind=args.dataset_kind,
            dataset_prefix=args.dataset_prefix,
            base_model_name=args.base_model_name,
            model_path=args.model_path,
            prompt_mode=args.prompt_mode,
            base_prompt=args.base_prompt,
            prompts_file=args.prompts_file,
            examples_path=args.examples_path,
            shots=args.shots,
            example_strategy=args.example_strategy,
            synth_per_label=args.synth_per_label,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_length=args.max_length,
            num_return_sequences=args.num_return_sequences,
        )
        print(json.dumps({"ok": True, "flow": "scheduled", "result": result["status"]}, indent=2))

    elif args.flow == "training":
        result = model_training_pipeline(
            mlflow_uri=args.mlflow_uri,
            log_level=args.log_level,
            seed=args.seed,
            training_data_path=args.training_data_path,
            validation_data_path=args.validation_data_path,
            base_model_name=args.base_model_name,
            experiment_name=args.experiment_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            base_prompt=args.base_prompt,
            num_samples=args.num_samples,
            max_length=args.max_length,
        )
        print(json.dumps({"ok": True, "flow": "training", "result": result["status"]}, indent=2))

    elif args.flow == "monitoring":
        result = monitoring_maintenance_flow(cleanup_days=args.cleanup_days)
        print(json.dumps({"ok": True, "flow": "monitoring", "result": result["status"], "health": result["monitoring"]["health_score"]}, indent=2))
