# # src/ingestion.py

"""
Data ingestion module for the synthetic data pipeline.
Handles loading, cleaning, and preparing real text data for training.
Optimized for AG News/IMDB datasets with instruction-tuning format for Llama models.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
import random
import tempfile

import pandas as pd
from datasets import load_dataset
import mlflow

logger = logging.getLogger(__name__)


@dataclass
class UnifiedItem:
    """Unified data structure for classification datasets"""
    text: str
    label: int
    
    def __post_init__(self):
        """Validate data after initialization"""
        if not isinstance(self.text, str):
            raise ValueError(f"text must be string, got {type(self.text)}")
        if not isinstance(self.label, int):
            raise ValueError(f"label must be int, got {type(self.label)}")


# Dataset-specific label mappings
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


# Instruction templates for GENERATION ONLY (not for training data)
GENERATION_TEMPLATES = {
    "ag_news": (
        "Now you are a journalist writing news articles. You are given a topic and must write a "
        "corresponding news article for it. You are also given a length requirement. You must "
        "ensure your news meets the length requirement.\n\n"
        "Can you write a news report with the topic {label_name}? The length requirement is: "
        "50 words. Please be creative and write unique news articles."
    ),
    "imdb": (
        "Now you are a movie critic. You need to have delicate emotions, unique perspectives, "
        "and a distinctive style. You are going to write a highly polar review for a movie and "
        "post it on IMDB. You are given a movie genre/style and a length requirement. You must "
        "come up with a movie that corresponds to the genre/style and write a review that meets "
        "the length requirement.\n\n"
        "Write a film review for a drama movie to express {label_name} feedback. Each review "
        "should have 80 words. Be sure to express your personal insights and feelings. "
        "Please be creative and write unique movie reviews."
    )
}


def _unify_ag_news(ds) -> Dict[str, List[UnifiedItem]]:
    """Convert AG News dataset to unified format"""
    def conv(ex):
        # AG News has 'text' and 'label' fields
        return UnifiedItem(text=ex["text"].strip(), label=int(ex["label"]))
    
    return {
        "train": [conv(x) for x in ds["train"]],
        "test": [conv(x) for x in ds["test"]],
    }


def _unify_imdb(ds) -> Dict[str, List[UnifiedItem]]:
    """Convert IMDB dataset to unified format"""
    def conv(ex):
        # IMDB has 'text' and 'label' fields
        return UnifiedItem(text=ex["text"].strip(), label=int(ex["label"]))
    
    return {
        "train": [conv(x) for x in ds["train"]],
        "test": [conv(x) for x in ds["test"]],
    }


def load_unified(
    dataset: str, 
    limit_per_split: Optional[int] = None
) -> Tuple[Dict[str, List[UnifiedItem]], Dict[int, str]]:
    """
    Load and unify dataset format.
    
    Args:
        dataset: Dataset name ('ag_news' or 'imdb')
        limit_per_split: Max samples per split (None = load all)
    
    Returns:
        Tuple of (unified_splits, label_map)
    """
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
        raise ValueError(f"Unsupported dataset: {dataset}. Must be 'ag_news' or 'imdb'")

    # Apply limit if specified
    if limit_per_split is not None:
        for split in uni:
            uni[split] = uni[split][:limit_per_split]
    
    logger.info(f"Loaded {dataset}: {', '.join([f'{k}={len(v)}' for k, v in uni.items()])}")
    return uni, label_map


def sample_seeds_per_class(
    items: List[UnifiedItem], 
    k_per_class: int, 
    num_classes: int,
    seed: int = 1234
) -> Dict[int, List[UnifiedItem]]:
    """
    Sample k examples per class for few-shot learning.
    
    Args:
        items: List of UnifiedItem objects
        k_per_class: Number of examples per class
        num_classes: Total number of classes
        seed: Random seed for reproducibility
    
    Returns:
        Dict mapping label -> list of sampled examples
    """
    # Group by class
    by_class: Dict[int, List[UnifiedItem]] = {i: [] for i in range(num_classes)}
    for item in items:
        if item.label in by_class:
            by_class[item.label].append(item)
    
    # Sample from each class
    seeds = {}
    rng = random.Random(seed)
    for class_id in range(num_classes):
        pool = by_class[class_id]
        if not pool:
            logger.warning(f"No examples found for class {class_id}")
            seeds[class_id] = []
            continue
        
        rng.shuffle(pool)
        seeds[class_id] = pool[:k_per_class]
        logger.info(f"Sampled {len(seeds[class_id])} examples for class {class_id}")
    
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
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataIngestion initialized: processed_dir={self.processed_dir}")
    
    def _load_hf_dataset(
        self,
        dataset_id: str,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> List[str]:
        """
        Load dataset from Hugging Face and return raw texts only.
        
        Args:
            dataset_id: HuggingFace dataset identifier
            split: Dataset split to load
            max_samples: Maximum number of samples to load
        
        Returns:
            List of text strings
        """
        logger.info(f"Loading {dataset_id} ({split} split)...")
        ds = load_dataset(dataset_id, split=split)
        
        texts: List[str] = []
        for i, row in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            
            text = (row.get("text") or "").strip()
            if text:
                texts.append(text)
        
        logger.info(f"Loaded {len(texts)} texts from {dataset_id}")
        return texts
    
    def _load_local_texts(
        self,
        source_path: str,
        max_samples: Optional[int] = None,
    ) -> List[str]:
        """
        Load texts from local file or directory.
        
        Supports:
        - Single file (.txt, .json, .jsonl)
        - Directory with multiple text files
        
        Args:
            source_path: Path to file or directory
            max_samples: Maximum samples to load
        
        Returns:
            List of text strings
        """
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Local source_path not found: {source_path}")

        texts: List[str] = []

        if path.is_file():
            texts = self._load_text_file(path)
        else:
            # Load from directory
            candidates = list(path.glob("train.*"))
            if not candidates:
                candidates = (
                    list(path.glob("*.txt")) + 
                    list(path.glob("*.jsonl")) + 
                    list(path.glob("*.json"))
                )

            for file_path in candidates:
                try:
                    texts.extend(self._load_text_file(file_path))
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue

        # Apply sample limit
        if max_samples is not None and len(texts) > max_samples:
            texts = texts[:max_samples]
        
        logger.info(f"Loaded {len(texts)} texts from {source_path}")
        return texts
    
    def _load_text_file(self, file_path: Path) -> List[str]:
        """
        Load text data from a single file.
        
        Args:
            file_path: Path to file
        
        Returns:
            List of text strings
        """
        texts = []
        
        if file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by paragraphs or double newlines
                texts = [text.strip() for text in content.split('\n\n') if text.strip()]
        
        elif file_path.suffix == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if isinstance(data, dict) and 'text' in data:
                        texts.append(data['text'])
                    elif isinstance(data, str):
                        texts.append(data)
        
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            texts.append(item)
                        elif isinstance(item, dict) and 'text' in item:
                            texts.append(item['text'])
                elif isinstance(data, dict) and 'texts' in data:
                    texts = data['texts']
        
        return texts
    
    def clean_and_validate_text(
        self, 
        texts: List[str], 
        min_length: int = 10,
        max_length: int = 2000  # Increased for AG News articles
    ) -> List[str]:
        """
        Clean and validate text samples.
        
        Args:
            texts: List of raw text strings
            min_length: Minimum character length
            max_length: Maximum character length
        
        Returns:
            List of cleaned and validated texts
        """
        logger.info(f"Cleaning {len(texts)} texts...")
        cleaned_texts = []
        stats = {
            'too_short': 0,
            'too_long': 0,
            'low_alpha': 0,
            'repetitive': 0,
            'valid': 0
        }
        
        for text in texts:
            text = text.strip()
            
            # Skip empty or very short texts
            if len(text) < min_length:
                stats['too_short'] += 1
                continue
            
            # Truncate very long texts at word boundary
            if len(text) > max_length:
                text = text[:max_length].rsplit(' ', 1)[0]
                stats['too_long'] += 1
            
            # Check alphabetic content ratio
            if len(text) > 0:
                alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
                if alpha_ratio < 0.5:  # At least 50% alphabetic or space
                    stats['low_alpha'] += 1
                    continue
            
            # Check for excessive repetition
            words = text.split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:  # Too repetitive
                    stats['repetitive'] += 1
                    continue
            
            cleaned_texts.append(text)
            stats['valid'] += 1
        
        logger.info(
            f"Cleaning complete: {stats['valid']} valid, "
            f"{stats['too_short']} too short, {stats['too_long']} truncated, "
            f"{stats['low_alpha']} low alpha, {stats['repetitive']} repetitive"
        )
        
        return cleaned_texts
    
    def _format_for_instruction_tuning(
        self, 
        items: List[UnifiedItem], 
        dataset_kind: str,
        include_label_in_output: bool = False
    ) -> List[str]:
        """
        Format data for fine-tuning (plain text with optional label prefix).
        
        NOTE: For LoRA fine-tuning, we use PLAIN TEXT (not instruction-formatted).
        The instruction templates are only used during GENERATION/INFERENCE.
        
        Args:
            items: List of UnifiedItem objects with text and labels
            dataset_kind: Dataset type ('ag_news' or 'imdb')
            include_label_in_output: Whether to include label prefix (e.g., "[World]")
        
        Returns:
            List of plain text strings (optionally with label prefix)
        """
        dataset_kind = dataset_kind.lower()
        if dataset_kind not in LABEL_MAPS:
            raise ValueError(f"Unknown dataset: {dataset_kind}")
        
        label_map = LABEL_MAPS[dataset_kind]
        
        formatted_texts = []
        for item in items:
            label_name = label_map[item.label]
            
            # For fine-tuning: use plain text, optionally with label prefix
            if include_label_in_output:
                # Format: [Label] Text
                formatted = f"[{label_name}] {item.text}"
            else:
                # Format: Just the plain text
                formatted = item.text
            
            formatted_texts.append(formatted)
        
        logger.info(
            f"Formatted {len(formatted_texts)} texts for fine-tuning "
            f"(label_prefix={'yes' if include_label_in_output else 'no'})"
        )
        return formatted_texts
    
    def create_dataset_splits(
        self, 
        texts: List[str], 
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42
    ) -> Dict[str, List[str]]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            texts: List of text samples
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            shuffle: Whether to shuffle before splitting
            seed: Random seed for shuffling
        
        Returns:
            Dict with 'train', 'validation', 'test' keys
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Shuffle if requested
        if shuffle:
            rng = random.Random(seed)
            texts = texts.copy()
            rng.shuffle(texts)
        
        n = len(texts)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits = {
            'train': texts[:train_end],
            'validation': texts[train_end:val_end],
            'test': texts[val_end:]
        }
        
        logger.info(
            f"Dataset splits - Train: {len(splits['train'])}, "
            f"Val: {len(splits['validation'])}, Test: {len(splits['test'])}"
        )
        
        return splits
    
    def save_processed_data(
        self, 
        data_splits: Dict[str, List[str]], 
        dataset_name: str, 
        version: str = "v1",
        save_metadata: bool = True
    ) -> Dict[str, Path]:
        """
        Save processed data splits to JSON files.
        
        Args:
            data_splits: Dict of split_name -> list of texts
            dataset_name: Name for the dataset
            version: Version identifier
            save_metadata: Whether to log to metadata database
        
        Returns:
            Dict of split_name -> file path
        """
        logger.info(f"Saving processed data for {dataset_name} {version}...")
        saved_paths = {}
        
        for split_name, texts in data_splits.items():
            # Create filename
            filename = f"{dataset_name}_{version}_{split_name}.json"
            file_path = self.processed_dir / filename
            
            # Prepare data structure
            data = {
                'dataset_name': dataset_name,
                'version': version,
                'split': split_name,
                'size': len(texts),
                'created_at': datetime.now().isoformat(),
                'texts': texts
            }
            
            # Save to JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            saved_paths[split_name] = file_path
            logger.info(f"Saved {split_name}: {len(texts)} samples -> {file_path}")
            
            # Log to metadata database
            if save_metadata:
                try:
                    metadata = {
                        'split': split_name,
                        'avg_length': sum(len(t) for t in texts) / len(texts) if texts else 0,
                        'vocab_size': len(set(' '.join(texts).split())) if texts else 0,
                        'processing_params': {
                            'min_length': 10,
                            'max_length': 2000
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
                    logger.debug(f"Logged to DB with ID: {dataset_id}")
                except Exception as e:
                    logger.warning(f"Failed to log to metadata DB: {e}")
        
        return saved_paths
    
    def log_to_mlflow(
        self, 
        data_splits: Dict[str, List[str]],
        dataset_name: str, 
        version: str
    ) -> None:
        """
        Log dataset information to MLflow.
        
        Args:
            data_splits: Dict of split_name -> list of texts
            dataset_name: Dataset name
            version: Dataset version
        """
        # Set up MLflow tracking
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            uri = f"file://{(Path.cwd() / 'mlruns').resolve()}"
            mlflow.set_tracking_uri(uri)

        os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
        mlflow.set_experiment("synthetic-lm-pipeline")

        total = sum(len(v) for v in data_splits.values())
        
        with mlflow.start_run(run_name=f"data_ingestion_{dataset_name}_{version}"):
            # Log parameters
            mlflow.log_param("dataset_name", dataset_name)
            mlflow.log_param("version", version)
            mlflow.log_param("total_samples", total)

            # Per-split parameters
            for split_name, texts in data_splits.items():
                mlflow.log_param(f"{split_name}_size", len(texts))
                if texts:
                    avg_len = sum(len(t) for t in texts) / len(texts)
                    mlflow.log_param(f"{split_name}_avg_length", round(avg_len, 2))

            # Build summary DataFrame
            records = []
            for split_name, texts in data_splits.items():
                for t in texts:
                    records.append({
                        "split": split_name, 
                        "text_len": len(t),
                        "word_count": len(t.split())
                    })
            
            if records:
                df = pd.DataFrame.from_records(records)

                # Log aggregate metrics
                mlflow.log_metric("total_chars", int(df["text_len"].sum()))
                mlflow.log_metric("avg_text_length", float(df["text_len"].mean()))
                mlflow.log_metric("avg_word_count", float(df["word_count"].mean()))
                
                # Count unique texts
                all_texts = [t for texts in data_splits.values() for t in texts]
                mlflow.log_metric("unique_texts", len(set(all_texts)))

                # Save summary as artifact
                with tempfile.TemporaryDirectory() as tmpd:
                    csv_path = Path(tmpd) / f"{dataset_name}_{version}_summary.csv"
                    df.to_csv(csv_path, index=False)
                    mlflow.log_artifact(str(csv_path), artifact_path="ingestion_summary")

            logger.info(f"[MLflow] Logged to {uri}")
    
    def run_ingestion_pipeline(
        self,
        source_type: str = "huggingface",
        source_path: Optional[str] = None,
        dataset_name: str = "ag_news",
        version: str = "v1",
        max_samples: Optional[int] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        format_for_instruction_tuning: bool = False,  # Changed default to False
        include_label_prefix: bool = False,  # New parameter
        shuffle: bool = True,
        seed: int = 42,
    ) -> Dict[str, Path]:
        """
        Main ingestion pipeline: load -> clean -> format -> split -> save.
        
        IMPORTANT: For LoRA fine-tuning, keep format_for_instruction_tuning=False.
        The model should be fine-tuned on PLAIN TEXT (the actual articles).
        Instruction templates are only used during GENERATION, not training.
        
        Args:
            source_type: 'huggingface' or 'file'
            source_path: HF dataset ID or local path (if None, uses dataset_name)
            dataset_name: Dataset identifier ('ag_news' or 'imdb')
            version: Version string
            max_samples: Max samples to load (None = all)
            train_ratio: Training split ratio
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
            format_for_instruction_tuning: [DEPRECATED] Use plain text for LoRA
            include_label_prefix: Whether to add "[Label] " prefix to text
            shuffle: Shuffle before splitting
            seed: Random seed
        
        Returns:
            Dict mapping split names to file paths
        """
        logger.info("=" * 60)
        logger.info("Starting data ingestion pipeline")
        logger.info(f"Dataset: {dataset_name}, Source: {source_type}")
        logger.info(f"Format: {'Plain text' if not format_for_instruction_tuning else 'DEPRECATED'}")
        logger.info("=" * 60)

        try:
            dataset_kind = dataset_name.lower().strip()
            
            # Validate dataset
            if dataset_kind not in LABEL_MAPS:
                raise ValueError(
                    f"Unsupported dataset: {dataset_kind}. "
                    f"Must be one of: {list(LABEL_MAPS.keys())}"
                )

            # Warn if using deprecated instruction formatting
            if format_for_instruction_tuning:
                logger.warning(
                    "⚠️  format_for_instruction_tuning=True is DEPRECATED for LoRA fine-tuning!"
                )
                logger.warning(
                    "⚠️  Use plain text (False) for training. "
                    "Instructions are only for generation."
                )

            if source_type == "huggingface":
                hf_id = source_path or dataset_kind
                
                # Always load with labels (needed for optional label prefix)
                splits, label_map = load_unified(hf_id, limit_per_split=max_samples)
                train_items = splits.get("train", [])
                
                if not train_items:
                    raise ValueError(f"No training data loaded from {hf_id}")
                
                logger.info(f"Loaded {len(train_items)} items with labels")

            elif source_type == "file":
                if not source_path:
                    raise ValueError("source_path required for source_type='file'")
                
                texts = self._load_local_texts(source_path, max_samples=max_samples)
                train_items = [UnifiedItem(text=t, label=0) for t in texts]
            
            else:
                raise ValueError(f"Unsupported source_type: {source_type}")

            logger.info(f"Cleaning {len(train_items)} items...")
            raw_texts = [item.text for item in train_items]
            cleaned_texts = self.clean_and_validate_text(raw_texts)

            if not cleaned_texts:
                raise ValueError("No valid texts after cleaning!")

            # Reconstruct items with cleaned texts
            cleaned_items = []
            for i, text in enumerate(cleaned_texts):
                if i < len(train_items):
                    cleaned_items.append(
                        UnifiedItem(text=text, label=train_items[i].label)
                    )
            
            if format_for_instruction_tuning and source_type == "huggingface":
                # DEPRECATED: Old instruction-formatting approach
                logger.warning("Using DEPRECATED instruction formatting")
                formatted_texts = self._format_for_instruction_tuning(
                    cleaned_items, 
                    dataset_kind,
                    include_label_in_output=include_label_prefix
                )
            elif include_label_prefix and source_type == "huggingface":
                # Add label prefix: "[World] Text..."
                logger.info("Adding label prefix to texts")
                label_map = LABEL_MAPS[dataset_kind]
                formatted_texts = [
                    f"[{label_map[item.label]}] {item.text}"
                    for item in cleaned_items
                ]
            else:
                # Plain text (RECOMMENDED for LoRA fine-tuning)
                logger.info("Using plain text format (recommended for LoRA)")
                formatted_texts = [item.text for item in cleaned_items]

            data_splits = self.create_dataset_splits(
                formatted_texts,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                shuffle=shuffle,
                seed=seed
            )

            saved_paths = self.save_processed_data(
                data_splits, 
                dataset_name, 
                version
            )

            # Normalize keys
            key_map = {
                "val": "validation",
                "valid": "validation",
                "validation": "validation",
                "train": "train",
                "test": "test",
            }
            normalized_paths = {}
            for k, v in saved_paths.items():
                nk = key_map.get(k, k)
                normalized_paths[nk] = Path(v) if not isinstance(v, Path) else v

            # -------------------------
            # 6) LOG TO MLFLOW
            # -------------------------
            try:
                self.log_to_mlflow(data_splits, dataset_name, version)
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

            logger.info("=" * 60)
            logger.info("Data ingestion pipeline completed successfully")
            logger.info(f"Format used: {'Plain text' if not format_for_instruction_tuning else 'Instruction-formatted'}")
            logger.info("=" * 60)
            
            return normalized_paths

        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    import sys
    import argparse
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    from src.pipeline_setup import Config, MetadataDB, setup_logging
    
    parser = argparse.ArgumentParser(
        description="Run data ingestion pipeline for synthetic text generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--source-type", 
        default="huggingface", 
        choices=["huggingface", "file"],
        help="Source type for data"
    )
    parser.add_argument(
        "--source-path",
        help="Path to data source (HF dataset ID or local path)"
    )
    parser.add_argument(
        "--dataset-name", 
        default="ag_news",
        choices=["ag_news", "imdb"],
        help="Dataset name"
    )
    parser.add_argument(
        "--version", 
        default="v1",
        help="Dataset version"
    )
    parser.add_argument(
        "--max-samples", 
        type=int,
        help="Maximum samples to process (default: all)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training split ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio"
    )
    parser.add_argument(
        "--no-instruction-format",
        action="store_true",
        help="Use plain text format (RECOMMENDED for LoRA fine-tuning)"
    )
    parser.add_argument(
        "--include-label-prefix",
        action="store_true",
        help="Add [Label] prefix to each text"
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before split"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
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
            max_samples=args.max_samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            format_for_instruction_tuning=False,  # Always False for LoRA
            include_label_prefix=args.include_label_prefix,
            shuffle=not args.no_shuffle,
            seed=args.seed,
        )
        
        print("\n" + "=" * 60)
        print("Ingestion completed successfully!")
        print("=" * 60)
        print("\nSaved files:")
        for split, path in saved_paths.items():
            file_size = path.stat().st_size / 1024 / 1024  # MB
            print(f"  {split:12s}: {path}")
            print(f"               Size: {file_size:.2f} MB")
        
        print("\n" + "=" * 60)
        print("Next steps:")
        print("  1. Inspect data: python scripts/inspect_data.py")
        print("  2. Run training: python pipelines/prefect/flows.py e2e")
        print("=" * 60)
            
    except Exception as e:
        logger.error(f" Pipeline failed: {e}")
        sys.exit(1)