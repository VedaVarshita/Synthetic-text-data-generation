# src/ingestion.py
"""
Data ingestion module for the synthetic data pipeline.
Handles loading, cleaning, and preparing real text data for training.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pandas as pd
from datasets import Dataset, load_dataset
import mlflow
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

import tempfile

logger = logging.getLogger(__name__)

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

try:
    from datasets import load_dataset
except Exception as e:
    raise ImportError("Requires the 'datasets' library. Install: pip install datasets") from e

@dataclass
class UnifiedItem:
    text: str
    label: int

LABEL_MAPS = {
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

def _unify_ag_news(ds) -> Dict[str, List[UnifiedItem]]:
    def conv(ex):
        return UnifiedItem(text=ex["text"], label=ex["label"])
    return {
        "train": [conv(x) for x in ds["train"]],
        "test": [conv(x) for x in ds["test"]],
    }

def _unify_imdb(ds) -> Dict[str, List[UnifiedItem]]:
    def conv(ex):
        return UnifiedItem(text=ex["text"], label=ex["label"])
    return {
        "train": [conv(x) for x in ds["train"]],
        "test": [conv(x) for x in ds["test"]],
    }

def load_unified(dataset: str, limit_seed: Optional[int] = None) -> Tuple[Dict[str, List[UnifiedItem]], Dict[int, str]]:
    dataset = dataset.lower()
    if dataset == "ag_news":
        ds = load_dataset("ag_news")
        uni = _unify_ag_news(ds)
        label_map = LABEL_MAPS["ag_news"]
    elif dataset == "imdb":
        ds = load_dataset("imdb")
        uni = _unify_imdb(ds)
        label_map = LABEL_MAPS["imdb"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if limit_seed is not None:
        for split in uni:
            uni[split] = uni[split][:limit_seed]
    return uni, label_map

def sample_seeds_per_class(items: List[UnifiedItem], k_per_class: int, num_classes: int) -> Dict[int, List[UnifiedItem]]:
    """Sample up to k_per_class seed examples per class from items."""
    by_class: Dict[int, List[UnifiedItem]] = {i: [] for i in range(num_classes)}
    for it in items:
        if it.label in by_class:
            by_class[it.label].append(it)
    seeds = {}
    rng = random.Random(1234)
    for c in range(num_classes):
        pool = by_class[c]
        rng.shuffle(pool)
        seeds[c] = pool[:k_per_class]
    return seeds



class DataIngestion:
    """Handles data ingestion and preprocessing for the pipeline"""
    
    def __init__(self, config: Dict[str, Any], metadata_db, data_dir: Path):
        self.config = config
        self.metadata_db = metadata_db
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        
        # Ensure directories exist
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
    
    # def load_sample_dataset(self, dataset_name: str = "wikitext", 
    #                       config_name: str = "wikitext-2-raw-v1",
    #                       split: str = "train",
    #                       max_samples: int = 1000) -> Dataset:
    #     """
    #     Load a sample dataset for demonstration purposes.
    #     Using WikiText-2 as it's free and good for language modeling.
    #     """
    #     logger.info(f"Loading {dataset_name} dataset...")
        
    #     try:
    #         # Load dataset from Hugging Face Hub
    #         dataset = load_dataset(dataset_name, config_name, split=split)
            
    #         # Take a subset for demonstration
    #         if max_samples and len(dataset) > max_samples:
    #             dataset = dataset.select(range(max_samples))
            
    #         logger.info(f"Loaded {len(dataset)} samples from {dataset_name}")
    #         return dataset
            
    #     except Exception as e:
    #         logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
    #         raise



    
    def load_sample_dataset(
        self,
        dataset_name: str,
        config_name: Optional[str] = None,  # <-- make this optional
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        # Only pass config if provided (wikitext etc.). AG/IMDB have no config_name.
        if config_name:
            ds = load_dataset(dataset_name, config_name, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)

        items = []
        for r in ds:
            txt = (r.get("text") or "").strip()
            if txt:
                row = {"text": txt}
                if "label" in r:
                    row["label"] = r["label"]
                items.append(row)

        if max_samples is not None:
            items = items[:max_samples]
        return items

    
    def load_custom_text_file(self, file_path: Union[str, Path]) -> List[str]:
        """Load text data from a custom file (txt, jsonl, etc.)"""
        file_path = Path(file_path)
        logger.info(f"Loading custom text file: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        texts = []
        
        if file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by paragraphs or double newlines
                texts = [text.strip() for text in content.split('\n\n') if text.strip()]
        
        elif file_path.suffix == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if isinstance(data, dict) and 'text' in data:
                        texts.append(data['text'])
                    elif isinstance(data, str):
                        texts.append(data)
        
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts.extend([item if isinstance(item, str) else item.get('text', '') 
                                for item in data])
                elif isinstance(data, dict) and 'texts' in data:
                    texts = data['texts']
        
        logger.info(f"Loaded {len(texts)} text samples from {file_path}")
        return texts
    
    def clean_and_validate_text(self, texts: List[str], 
                              min_length: int = 10,
                              max_length: int = 1000) -> List[str]:
        """Clean and validate text samples"""
        logger.info("Cleaning and validating text data...")
        
        cleaned_texts = []
        
        for text in texts:
            # Basic cleaning
            text = text.strip()
            
            # Skip empty or very short texts
            if len(text) < min_length:
                continue
            
            # Truncate very long texts
            if len(text) > max_length:
                text = text[:max_length].rsplit(' ', 1)[0]  # Cut at word boundary
            
            # Remove texts with too many special characters
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
            if alpha_ratio < 0.7:  # At least 70% alphabetic or space
                continue
            
            # Basic profanity/content filtering could go here
            # For now, just check for excessive repetition
            words = text.split()
            if len(set(words)) / len(words) < 0.3:  # Too repetitive
                continue
            
            cleaned_texts.append(text)
        
        logger.info(f"Cleaned dataset: {len(cleaned_texts)} valid texts "
                   f"(filtered out {len(texts) - len(cleaned_texts)})")
        
        return cleaned_texts
    
    def create_dataset_splits(self, texts: List[str], 
                            train_ratio: float = 0.8,
                            val_ratio: float = 0.1,
                            test_ratio: float = 0.1) -> Dict[str, List[str]]:
        """Split dataset into train/val/test"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        n = len(texts)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits = {
            'train': texts[:train_end],
            'validation': texts[train_end:val_end],
            'test': texts[val_end:]
        }
        
        logger.info(f"Dataset splits - Train: {len(splits['train'])}, "
                   f"Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
        
        return splits
    
    def save_processed_data(self, data_splits: Dict[str, List[str]], 
                          dataset_name: str, version: str = "v1") -> Dict[str, Path]:
        """Save processed data to files and log to database"""
        logger.info(f"Saving processed data for {dataset_name} {version}...")
        
        saved_paths = {}
        
        for split_name, texts in data_splits.items():
            # Create filename
            filename = f"{dataset_name}_{version}_{split_name}.json"
            file_path = self.processed_dir / filename
            
            # Save as JSON for easy loading
            data = {
                'dataset_name': dataset_name,
                'version': version,
                'split': split_name,
                'size': len(texts),
                'created_at': datetime.now().isoformat(),
                'texts': texts
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            saved_paths[split_name] = file_path
            
            # Log to metadata database
            metadata = {
                'split': split_name,
                'avg_length': sum(len(t) for t in texts) / len(texts),
                'vocab_size': len(set(' '.join(texts).split())),
                'processing_params': {
                    'min_length': 10,
                    'max_length': 1000
                }
            }
            
            dataset_id = self.metadata_db.log_dataset(
                name=f"{dataset_name}_{split_name}",
                version=version,
                type_="real",
                size=len(texts),
                file_path=str(file_path),
                metadata=metadata
            )
            
            logger.info(f"Saved {split_name} split to {file_path} (DB ID: {dataset_id})")
        
        return saved_paths
    
    # def log_to_mlflow(self, data_splits: Dict[str, List[str]], 
    #                  dataset_name: str, version: str):
    #     """Log dataset information to MLflow"""
    #     with mlflow.start_run(run_name=f"data_ingestion_{dataset_name}_{version}"):
    #         # Log parameters
    #         mlflow.log_param("dataset_name", dataset_name)
    #         mlflow.log_param("version", version)
    #         mlflow.log_param("total_samples", sum(len(split) for split in data_splits.values()))
            
    #         for split_name, texts in data_splits.items():
    #             mlflow.log_param(f"{split_name}_size", len(texts))
    #             mlflow.log_param(f"{split_name}_avg_length", 
    #                            sum(len(t) for t in texts) / len(texts))
            
    #         # Convert to pandas for MLflow dataset logging
    #         all_texts = []
    #         all_splits = []
    #         for split_name, texts in data_splits.items():
    #             all_texts.extend(texts)
    #             all_splits.extend([split_name] * len(texts))
            
    #         df = pd.DataFrame({
    #             'text': all_texts,
    #             'split': all_splits
    #         })
            
    #         # Log as MLflow dataset
    #         dataset = mlflow.data.from_pandas(df, source=dataset_name)
    #         mlflow.log_input(dataset, context="training_data")
            
    #         # Log dataset statistics
    #         mlflow.log_metric("total_chars", df['text'].str.len().sum())
    #         mlflow.log_metric("avg_text_length", df['text'].str.len().mean())
    #         mlflow.log_metric("unique_texts", df['text'].nunique())
            
    #         logger.info(f"Dataset {dataset_name} logged to MLflow")

    def log_to_mlflow(self, data_splits: Dict[str, List[str]],
                    dataset_name: str, version: str) -> None:
        """Log dataset information to MLflow (robust, no dataset-source plugins)."""

        # Pin tracking URI to local file store if not already set
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            uri = f"file://{(Path.cwd() / 'mlruns').resolve()}"
            mlflow.set_tracking_uri(uri)

        # Optional: avoid extra system metrics collection (sometimes slow on local dev)
        os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")

        mlflow.set_experiment("synthetic-lm-pipeline")

        # Flatten splits for metrics/artifacts
        total = sum(len(v) for v in data_splits.values())
        with mlflow.start_run(run_name=f"data_ingestion_{dataset_name}_{version}"):
            # Params
            mlflow.log_param("dataset_name", dataset_name)
            mlflow.log_param("version", version)
            mlflow.log_param("total_samples", total)

            # Per-split params & metrics
            for split_name, texts in data_splits.items():
                mlflow.log_param(f"{split_name}_size", len(texts))
                avg_len = (sum(len(t) for t in texts) / len(texts)) if texts else 0.0
                mlflow.log_param(f"{split_name}_avg_length", avg_len)

            # Build a small summary table and log as artifact (CSV)
            records = []
            for split_name, texts in data_splits.items():
                for t in texts:
                    records.append({"split": split_name, "text_len": len(t)})
            df = pd.DataFrame.from_records(records)

            # Simple aggregate metrics
            mlflow.log_metric("total_chars", int(df["text_len"].sum()) if not df.empty else 0)
            mlflow.log_metric("avg_text_length", float(df["text_len"].mean()) if not df.empty else 0.0)
            # (Unique texts) – cheap proxy without hashing
            uniq = 0
            if data_splits:
                uniq = len(set(sum((texts for texts in data_splits.values()), [])))
            mlflow.log_metric("unique_texts", int(uniq))

            # Write artifact safely
            with tempfile.TemporaryDirectory() as tmpd:
                csv_path = Path(tmpd) / f"{dataset_name}_{version}_summary.csv"
                df.to_csv(csv_path, index=False)
                mlflow.log_artifact(str(csv_path), artifact_path="ingestion_summary")

            # Done
            logger.info(f"[MLflow] Logged ingestion for {dataset_name}:{version} to {uri}")

    

    def run_ingestion_pipeline(
        self,
        source_type: str = "huggingface",
        source_path: str = "wikitext",
        dataset_name: str = "sample_text",
        version: str = "v1",
        max_samples: int = 1000,
        # allow explicit ratios if your create_dataset_splits accepts them
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Dict[str, Path]:
        """
        Unified entry point used by Prefect flows.
        Routes AG News/IMDB to the unified loader; everything else to generic loaders.
        Then: clean -> split -> save -> log.
        Returns dict of split -> file paths.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Starting data ingestion pipeline...")

        try:
            src = (source_path or "").lower()

            # 1) Load --------------------------------------------------------------
            if source_type == "huggingface" and src in {"ag_news", "imdb"}:
                # Use your unified path that returns {"train":[UnifiedItem], "test":[...], ...}
                # NOTE: implement self.load_unified(dataset_name: str, max_total: Optional[int]) -> Dict[str, List[UnifiedItem]]
                splits, _label_map = load_unified(src, limit_seed=max_samples)
                train_items = splits.get("train", [])
                texts: List[str] = [u.text for u in train_items if getattr(u, "text", "").strip()]

            elif source_type == "file":
                # Your existing file loader: returns List[str]
                texts = self.load_custom_text_file(source_path)

            elif source_type == "huggingface":
                # Generic HF datasets (e.g., wikitext). Make sure your load_sample_dataset accepts config_name=None.
                dataset = self.load_sample_dataset(
                    dataset_name=source_path,
                    config_name=None,
                    split="train",
                    max_samples=max_samples,
                )
                texts = [row.get("text", "").strip() for row in dataset if row.get("text")]

            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            # 2) Clean & validate --------------------------------------------------
            # Provide either clean_and_validate_text or fallback to your clean_texts + drop empties
            if hasattr(self, "clean_and_validate_text"):
                cleaned_texts = self.clean_and_validate_text(texts)
            else:
                cleaned = self.clean_texts(texts)
                cleaned_texts = [t for t in cleaned if t and t.strip()]

            if not cleaned_texts:
                raise ValueError("No valid texts after cleaning!")

            # 3) Split -------------------------------------------------------------
            # If your create_dataset_splits signature accepts ratios, pass them; otherwise call without.
            try:
                data_splits = self.create_dataset_splits(
                    cleaned_texts,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                )
            except TypeError:
                # fallback for older signature
                data_splits = self.create_dataset_splits(cleaned_texts)

            # 4) Save --------------------------------------------------------------
            saved_paths = self.save_processed_data(data_splits, dataset_name, version)

            # 5) Log to MLflow -----------------------------------------------------
            try:
                self.log_to_mlflow(data_splits, dataset_name, version)
            except Exception:
                logger.debug("MLflow logging skipped or failed; continuing.")

            logger.info("Data ingestion pipeline completed successfully.")
            # Ensure we return {split: Path}
            return {k: (Path(v) if not isinstance(v, Path) else v) for k, v in saved_paths.items()}

        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {e}")
            raise

# Example usage and CLI interface
if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    from src.pipeline_setup import Config, MetadataDB, setup_logging
    
    parser = argparse.ArgumentParser(description="Run data ingestion pipeline")
    parser.add_argument("--source-type", default="huggingface", 
                       choices=["huggingface", "file"],
                       help="Source type for data")
    parser.add_argument("--source-path", default="wikitext",
                       help="Path to data source")
    parser.add_argument("--dataset-name", default="sample_text",
                       help="Name for the dataset")
    parser.add_argument("--version", default="v1",
                       help="Dataset version")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="Maximum samples to process")
    
    args = parser.parse_args()
    
    # Initialize components
    config = Config()
    logger = setup_logging(config.logs_dir)
    metadata_db = MetadataDB(config.data_dir / "pipeline.db")
    
    # Run ingestion
    ingestion = DataIngestion(config.model_configs, metadata_db, config.data_dir)

    
    try:
        saved_paths = ingestion.run_ingestion_pipeline(
            source_type=args.source_type,
            source_path=args.source_path,
            dataset_name=args.dataset_name,
            version=args.version,
            max_samples=args.max_samples
        )
        
        print("\n✅ Ingestion completed! Saved files:")
        for split, path in saved_paths.items():
            print(f"  {split}: {path}")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
