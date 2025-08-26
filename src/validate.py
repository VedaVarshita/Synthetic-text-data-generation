# src/validate.py
"""
Comprehensive validation module for synthetic text data.
Includes quality, diversity, privacy, and similarity checks.
"""

import os
import json
import logging
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import mlflow
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)


from typing import List, Dict
# Simple quality metrics + optional DCScore

def dcscore(texts: List[str], embedder_name: str = "princeton-nlp/unsup-simcse-bert-base-uncased", temperature: float = 1.0) -> float:
    """
    DCScore: compute embeddings -> cosine similarity matrix -> softmax row-wise with temperature -> sum diagonal.
    Install deps: pip install sentence-transformers torch
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except Exception as e:
        raise ImportError("DCScore requires 'sentence-transformers' and 'torch'.") from e

    model = SentenceTransformer(embedder_name)
    with torch.no_grad():
        embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        sim = embs @ embs.T  # cosine since normalized
        sim = sim / temperature
        P = torch.softmax(sim, dim=1)
        score = torch.trace(P).item()
    return float(score)

def distinct_n(texts: List[str], n: int = 2) -> float:
    ngrams = []
    for t in texts:
        toks = t.strip().split()
        ngrams.extend([" ".join(toks[i:i+n]) for i in range(0, max(0, len(toks)-n+1))])
    denom = max(1, len(ngrams))
    return len(set(ngrams)) / denom

def self_bleu(texts: List[str], sample: int = 100) -> float:
    """Approx self-BLEU by sampling; lower is better (more diverse)."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except Exception as e:
        raise ImportError("self_bleu requires nltk. Install: pip install nltk") from e
    import random
    rng = random.Random(1337)
    if len(texts) <= 1:
        return 0.0
    sample = min(sample, len(texts)-1)
    scores = []
    ref_pool = texts[:]
    for _ in range(sample):
        hyp = rng.choice(texts)
        refs = [rng.choice(ref_pool).split() for _ in range(5)]
        score = sentence_bleu(refs, hyp.split(), smoothing_function=SmoothingFunction().method1)
        scores.append(score)
    return sum(scores)/len(scores)

def summarize_quality(texts: List[str], compute_dc: bool = False) -> Dict[str, float]:
    out = {
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
    }
    try:
        out["self_bleu"] = self_bleu(texts)
    except Exception:
        out["self_bleu"] = float("nan")
    if compute_dc:
        try:
            out["dcscore"] = dcscore(texts)
        except Exception:
            out["dcscore"] = float("nan")
    return out


class TextValidator:
    """Comprehensive text validation for synthetic datasets"""
    
    def __init__(self, config: Dict[str, Any], metadata_db):
        self.config = config
        self.metadata_db = metadata_db
        self.thresholds = config.get('validation_thresholds', {})
        
        # Initialize models
        self.embedding_model = None
        self.perplexity_model = None
        self.perplexity_tokenizer = None
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load embedding model
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load sentence embedding model for semantic similarity"""
        model_name = self.config.get('embeddings', {}).get('model_name', 'all-MiniLM-L6-v2')
        logger.info(f"Loading embedding model: {model_name}")
        
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def _load_perplexity_model(self, model_name: str = "distilgpt2"):
        """Load model for perplexity calculation"""
        if self.perplexity_model is None:
            logger.info(f"Loading perplexity model: {model_name}")
            
            try:
                self.perplexity_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.perplexity_model = AutoModelForCausalLM.from_pretrained(model_name)
                self.perplexity_model.to(self.device)
                self.perplexity_model.eval()
                
                if self.perplexity_tokenizer.pad_token is None:
                    self.perplexity_tokenizer.pad_token = self.perplexity_tokenizer.eos_token
                
                logger.info("Perplexity model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading perplexity model: {str(e)}")
                raise
    
    def calculate_perplexity(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """Calculate perplexity for a list of texts"""
        self._load_perplexity_model()
        
        logger.info(f"Calculating perplexity for {len(texts)} texts...")
        perplexities = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.perplexity_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Calculate loss (negative log likelihood)
                outputs = self.perplexity_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                # Convert to perplexity
                batch_perplexity = torch.exp(loss).item()
                perplexities.extend([batch_perplexity] * len(batch_texts))
        
        logger.info(f"Average perplexity: {np.mean(perplexities):.2f}")
        return perplexities
    
    def calculate_embedding_similarity(self, real_texts: List[str], 
                                     synthetic_texts: List[str]) -> Dict[str, float]:
        """Calculate semantic similarity between real and synthetic texts"""
        logger.info("Calculating embedding similarity...")
        
        # Encode texts
        real_embeddings = self.embedding_model.encode(real_texts, convert_to_tensor=True)
        synthetic_embeddings = self.embedding_model.encode(synthetic_texts, convert_to_tensor=True)
        
        # Calculate various similarity metrics
        results = {}
        
        # Mean embedding similarity
        real_mean = real_embeddings.mean(dim=0)
        synthetic_mean = synthetic_embeddings.mean(dim=0)
        results['mean_cosine_similarity'] = util.cos_sim(real_mean, synthetic_mean).item()
        
        # Distribution similarity (compare embedding distributions)
        # Use maximum mean discrepancy (MMD) approximation
        real_np = real_embeddings.cpu().numpy()
        synthetic_np = synthetic_embeddings.cpu().numpy()
        
        # Calculate pairwise similarities within and between sets
        real_real_sim = np.mean([
            cosine_similarity([real_np[i]], [real_np[j]])[0, 0] 
            for i in range(min(100, len(real_np))) 
            for j in range(i+1, min(100, len(real_np)))
        ])
        
        synth_synth_sim = np.mean([
            cosine_similarity([synthetic_np[i]], [synthetic_np[j]])[0, 0]
            for i in range(min(100, len(synthetic_np)))
            for j in range(i+1, min(100, len(synthetic_np)))
        ])
        
        real_synth_sim = np.mean([
            cosine_similarity([real_np[i]], [synthetic_np[j]])[0, 0]
            for i in range(min(50, len(real_np)))
            for j in range(min(50, len(synthetic_np)))
        ])
        
        results['distribution_similarity'] = 2 * real_synth_sim - real_real_sim - synth_synth_sim
        results['real_internal_similarity'] = real_real_sim
        results['synthetic_internal_similarity'] = synth_synth_sim
        results['cross_similarity'] = real_synth_sim
        
        logger.info(f"Embedding similarity results: {results}")
        return results
    
    def calculate_diversity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """Calculate diversity metrics for text dataset"""
        logger.info("Calculating diversity metrics...")
        
        results = {}
        
        # Basic uniqueness
        unique_texts = set(texts)
        results['uniqueness_ratio'] = len(unique_texts) / len(texts)
        
        # Vocabulary diversity
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        unique_words = set(all_words)
        results['vocabulary_size'] = len(unique_words)
        results['total_words'] = len(all_words)
        results['vocab_text_ratio'] = len(unique_words) / len(texts)
        
        # Average text length
        lengths = [len(text) for text in texts]
        results['avg_length'] = np.mean(lengths)
        results['length_std'] = np.std(lengths)
        
        # N-gram diversity
        results.update(self._calculate_ngram_diversity(texts))
        
        # Self-BLEU (diversity within generated texts)
        results['self_bleu'] = self._calculate_self_bleu(texts)
        
        logger.info(f"Diversity metrics: {results}")
        return results
    
    def _calculate_ngram_diversity(self, texts: List[str], max_n: int = 4) -> Dict[str, float]:
        """Calculate n-gram diversity metrics"""
        results = {}
        
        for n in range(1, max_n + 1):
            all_ngrams = []
            for text in texts:
                words = text.lower().split()
                if len(words) >= n:
                    text_ngrams = list(ngrams(words, n))
                    all_ngrams.extend(text_ngrams)
            
            if all_ngrams:
                unique_ngrams = set(all_ngrams)
                results[f'ngram_{n}_diversity'] = len(unique_ngrams) / len(all_ngrams)
            else:
                results[f'ngram_{n}_diversity'] = 0.0
        
        return results
    
    def _calculate_self_bleu(self, texts: List[str], sample_size: int = 100) -> float:
        """Calculate Self-BLEU score for diversity measurement"""
        if len(texts) < 2:
            return 0.0
        
        # Sample texts if too many
        if len(texts) > sample_size:
            texts = np.random.choice(texts, sample_size, replace=False).tolist()
        
        bleu_scores = []
        smoothing = SmoothingFunction().method1
        
        for i, text in enumerate(texts):
            reference_texts = texts[:i] + texts[i+1:]  # All other texts as references
            
            # Tokenize
            hypothesis = text.lower().split()
            references = [ref.lower().split() for ref in reference_texts[:10]]  # Limit references
            
            if references and hypothesis:
                try:
                    bleu_score = sentence_bleu(references, hypothesis, 
                                             weights=(0.25, 0.25, 0.25, 0.25),
                                             smoothing_function=smoothing)
                    bleu_scores.append(bleu_score)
                except:
                    continue
        
        return np.mean(bleu_scores) if bleu_scores else 0.0
    
    def check_privacy_violations(self, real_texts: List[str], 
                               synthetic_texts: List[str]) -> Dict[str, Any]:
        """Check for privacy violations in synthetic data"""
        logger.info("Checking privacy violations...")
        
        results = {
            'exact_duplicates': 0,
            'near_duplicates': 0,
            'duplicate_ratio': 0.0,
            'privacy_safe': True,
            'violations': []
        }
        
        real_texts_set = set(real_texts)
        
        # Check for exact duplicates
        exact_duplicates = []
        for i, synthetic_text in enumerate(synthetic_texts):
            if synthetic_text in real_texts_set:
                exact_duplicates.append({
                    'synthetic_index': i,
                    'text': synthetic_text[:100] + '...' if len(synthetic_text) > 100 else synthetic_text
                })
        
        results['exact_duplicates'] = len(exact_duplicates)
        results['duplicate_ratio'] = len(exact_duplicates) / len(synthetic_texts)
        results['violations'].extend(exact_duplicates)
        
        # Check for near duplicates using n-gram overlap
        near_duplicates = self._find_near_duplicates(real_texts, synthetic_texts)
        results['near_duplicates'] = len(near_duplicates)
        results['violations'].extend(near_duplicates)
        
        # Privacy assessment
        max_duplicate_ratio = self.thresholds.get('max_exact_duplicates', 0.05)
        results['privacy_safe'] = results['duplicate_ratio'] <= max_duplicate_ratio
        
        logger.info(f"Privacy check: {results['exact_duplicates']} exact duplicates, "
                   f"{results['near_duplicates']} near duplicates")
        
        return results
    
    def _find_near_duplicates(self, real_texts: List[str], 
                            synthetic_texts: List[str],
                            threshold: float = 0.8) -> List[Dict]:
        """Find near duplicates using TF-IDF similarity"""
        near_duplicates = []
        
        # Use TF-IDF for efficient similarity computation
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, stop_words='english')
        
        try:
            # Fit on real texts
            real_vectors = vectorizer.fit_transform(real_texts)
            synthetic_vectors = vectorizer.transform(synthetic_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(synthetic_vectors, real_vectors)
            
            # Find near duplicates
            for i, sim_scores in enumerate(similarities):
                max_sim = np.max(sim_scores)
                if max_sim > threshold:
                    max_idx = np.argmax(sim_scores)
                    near_duplicates.append({
                        'synthetic_index': i,
                        'real_index': max_idx,
                        'similarity': float(max_sim),
                        'synthetic_text': synthetic_texts[i][:100] + '...',
                        'real_text': real_texts[max_idx][:100] + '...'
                    })
        
        except Exception as e:
            logger.warning(f"Error in near duplicate detection: {str(e)}")
        
        return near_duplicates
    
    def validate_dataset(self, synthetic_texts: List[str],
                        real_texts: List[str] = None,
                        dataset_name: str = "synthetic_dataset") -> Dict[str, Any]:
        """Run comprehensive validation on synthetic dataset"""
        logger.info(f"Starting comprehensive validation for {dataset_name}...")
        
        validation_results = {
            'dataset_name': dataset_name,
            'total_samples': len(synthetic_texts),
            'timestamp': pd.Timestamp.now().isoformat(),
            'validations': {},
            'overall_passed': True,
            'failed_checks': []
        }
        
        # 1. Quality metrics (perplexity)
        try:
            perplexities = self.calculate_perplexity(synthetic_texts)
            avg_perplexity = np.mean(perplexities)
            
            quality_results = {
                'average_perplexity': avg_perplexity,
                'perplexity_std': np.std(perplexities),
                'perplexity_scores': perplexities[:10]  # Sample for logging
            }
            
            # Check thresholds
            min_perplexity = self.thresholds.get('min_avg_perplexity', 5.0)
            max_perplexity = self.thresholds.get('max_avg_perplexity', 100.0)
            
            quality_passed = min_perplexity <= avg_perplexity <= max_perplexity
            quality_results['passed'] = quality_passed
            
            if not quality_passed:
                validation_results['failed_checks'].append('perplexity')
                validation_results['overall_passed'] = False
            
            validation_results['validations']['quality'] = quality_results
            
        except Exception as e:
            logger.error(f"Quality validation failed: {str(e)}")
            validation_results['validations']['quality'] = {'error': str(e), 'passed': False}
            validation_results['overall_passed'] = False
        
        # 2. Diversity metrics
        try:
            diversity_results = self.calculate_diversity_metrics(synthetic_texts)
            
            # Check diversity thresholds
            min_uniqueness = self.thresholds.get('min_uniqueness_ratio', 0.7)
            diversity_passed = diversity_results['uniqueness_ratio'] >= min_uniqueness
            diversity_results['passed'] = diversity_passed
            
            if not diversity_passed:
                validation_results['failed_checks'].append('diversity')
                validation_results['overall_passed'] = False
            
            validation_results['validations']['diversity'] = diversity_results
            
        except Exception as e:
            logger.error(f"Diversity validation failed: {str(e)}")
            validation_results['validations']['diversity'] = {'error': str(e), 'passed': False}
            validation_results['overall_passed'] = False
        
        # 3. Privacy checks (if real texts provided)
        if real_texts:
            try:
                privacy_results = self.check_privacy_violations(real_texts, synthetic_texts)
                privacy_passed = privacy_results['privacy_safe']
                privacy_results['passed'] = privacy_passed
                
                if not privacy_passed:
                    validation_results['failed_checks'].append('privacy')
                    validation_results['overall_passed'] = False
                
                validation_results['validations']['privacy'] = privacy_results
                
            except Exception as e:
                logger.error(f"Privacy validation failed: {str(e)}")
                validation_results['validations']['privacy'] = {'error': str(e), 'passed': False}
                validation_results['overall_passed'] = False
        
        # 4. Semantic similarity (if real texts provided)
        if real_texts:
            try:
                similarity_results = self.calculate_embedding_similarity(real_texts, synthetic_texts)
                
                # Check similarity thresholds
                min_similarity = self.thresholds.get('min_embedding_similarity', 0.3)
                similarity_passed = similarity_results['mean_cosine_similarity'] >= min_similarity
                similarity_results['passed'] = similarity_passed
                
                if not similarity_passed:
                    validation_results['failed_checks'].append('similarity')
                    validation_results['overall_passed'] = False
                
                validation_results['validations']['similarity'] = similarity_results
                
            except Exception as e:
                logger.error(f"Similarity validation failed: {str(e)}")
                validation_results['validations']['similarity'] = {'error': str(e), 'passed': False}
                validation_results['overall_passed'] = False
        
        # Log results to MLflow
        self._log_validation_to_mlflow(validation_results)
        
        # Log to database
        if hasattr(self, 'metadata_db'):
            try:
                # This would need a dataset_id - in practice, you'd pass this in
                # For now, we'll log the validation results as metadata
                pass
            except Exception as e:
                logger.warning(f"Could not log to database: {str(e)}")
        
        logger.info(f"Validation completed. Overall passed: {validation_results['overall_passed']}")
        return validation_results
    
    def _log_validation_to_mlflow(self, results: Dict[str, Any]):
        """Log validation results to MLflow"""
        try:
            # # Set experiment name - this will create it if it doesn't exist
            # mlflow.set_experiment("synthetic-lm-pipeline")
            
            with mlflow.start_run(run_name=f"validation_{results['dataset_name']}"):
                # Log basic metrics
                mlflow.log_param("dataset_name", results['dataset_name'])
                mlflow.log_param("total_samples", results['total_samples'])
                mlflow.log_param("overall_passed", results['overall_passed'])
                
                # Log detailed metrics
                for validation_type, validation_data in results['validations'].items():
                    if isinstance(validation_data, dict) and 'error' not in validation_data:
                        # Log numeric metrics
                        for key, value in validation_data.items():
                            if isinstance(value, (int, float)) and key != 'passed':
                                mlflow.log_metric(f"{validation_type}_{key}", value)
                        
                        # Log pass/fail status
                        mlflow.log_param(f"{validation_type}_passed", validation_data.get('passed', False))
                
                # Log failed checks
                mlflow.log_param("failed_checks", ','.join(results['failed_checks']))
                
                logger.info("Validation results logged to MLflow")
                
        except Exception as e:
            logger.warning(f"Could not log to MLflow: {str(e)}")
    
    def generate_validation_report(self, results: Dict[str, Any], 
                                 output_path: Path = None) -> str:
        """Generate a human-readable validation report"""
        
        report_lines = [
            f"# Validation Report for {results['dataset_name']}",
            f"Generated: {results['timestamp']}",
            f"Total Samples: {results['total_samples']}",
            f"Overall Status: {'✅ PASSED' if results['overall_passed'] else '❌ FAILED'}",
            ""
        ]
        
        if results['failed_checks']:
            report_lines.extend([
                "## Failed Checks:",
                ", ".join(results['failed_checks']),
                ""
            ])
        
        # Detailed results
        for validation_type, validation_data in results['validations'].items():
            if isinstance(validation_data, dict) and 'error' not in validation_data:
                status = "✅ PASSED" if validation_data.get('passed', False) else "❌ FAILED"
                report_lines.extend([
                    f"## {validation_type.title()} Validation: {status}",
                    ""
                ])
                
                # Add specific metrics
                if validation_type == 'quality':
                    report_lines.extend([
                        f"- Average Perplexity: {validation_data.get('average_perplexity', 'N/A'):.2f}",
                        f"- Perplexity Std: {validation_data.get('perplexity_std', 'N/A'):.2f}",
                        ""
                    ])
                
                elif validation_type == 'diversity':
                    report_lines.extend([
                        f"- Uniqueness Ratio: {validation_data.get('uniqueness_ratio', 'N/A'):.3f}",
                        f"- Vocabulary Size: {validation_data.get('vocabulary_size', 'N/A')}",
                        f"- Self-BLEU Score: {validation_data.get('self_bleu', 'N/A'):.3f}",
                        f"- Average Length: {validation_data.get('avg_length', 'N/A'):.1f}",
                        ""
                    ])
                
                elif validation_type == 'privacy':
                    report_lines.extend([
                        f"- Exact Duplicates: {validation_data.get('exact_duplicates', 'N/A')}",
                        f"- Duplicate Ratio: {validation_data.get('duplicate_ratio', 'N/A'):.3f}",
                        f"- Near Duplicates: {validation_data.get('near_duplicates', 'N/A')}",
                        ""
                    ])
                
                elif validation_type == 'similarity':
                    report_lines.extend([
                        f"- Mean Cosine Similarity: {validation_data.get('mean_cosine_similarity', 'N/A'):.3f}",
                        f"- Cross Similarity: {validation_data.get('cross_similarity', 'N/A'):.3f}",
                        ""
                    ])
            
            elif 'error' in validation_data:
                report_lines.extend([
                    f"## {validation_type.title()} Validation: ❌ ERROR",
                    f"Error: {validation_data['error']}",
                    ""
                ])
        
        report = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            output_path.parent.mkdir(exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Validation report saved to {output_path}")
        
        return report


class ValidationPipeline:
    """End-to-end validation pipeline"""
    
    def __init__(self, config: Dict[str, Any], metadata_db):
        self.validator = TextValidator(config, metadata_db)
        self.metadata_db = metadata_db
    
    def validate_synthetic_dataset(self, synthetic_dataset_path: Path,
                                 real_dataset_path: Path = None,
                                 output_dir: Path = None) -> Dict[str, Any]:
        """Run validation pipeline on a synthetic dataset"""
        
        logger.info(f"Loading synthetic dataset from {synthetic_dataset_path}")
        
        # Load synthetic dataset
        with open(synthetic_dataset_path, 'r', encoding='utf-8') as f:
            synthetic_data = json.load(f)
        
        synthetic_texts = synthetic_data.get('texts', [])
        dataset_name = synthetic_data.get('dataset_name', 'unknown')
        
        # Load real dataset if provided
        real_texts = None
        if real_dataset_path and real_dataset_path.exists():
            logger.info(f"Loading real dataset from {real_dataset_path}")
            with open(real_dataset_path, 'r', encoding='utf-8') as f:
                real_data = json.load(f)
            real_texts = real_data.get('texts', [])
        
        # Run validation
        results = self.validator.validate_dataset(
            synthetic_texts=synthetic_texts,
            real_texts=real_texts,
            dataset_name=dataset_name
        )
        
        # Generate report
        if output_dir:
            output_dir.mkdir(exist_ok=True)
            report_path = output_dir / f"{dataset_name}_validation_report.md"
            report = self.validator.generate_validation_report(results, report_path)
        else:
            report = self.validator.generate_validation_report(results)
        
        return results
    
    def batch_validate_datasets(self, datasets_dir: Path,
                              real_dataset_path: Path = None,
                              output_dir: Path = None) -> Dict[str, Dict[str, Any]]:
        """Validate multiple synthetic datasets"""
        
        results = {}
        
        # Find all synthetic dataset files
        dataset_files = list(datasets_dir.glob("*_synthetic.json"))
        
        logger.info(f"Found {len(dataset_files)} synthetic datasets to validate")
        
        for dataset_file in dataset_files:
            try:
                dataset_results = self.validate_synthetic_dataset(
                    synthetic_dataset_path=dataset_file,
                    real_dataset_path=real_dataset_path,
                    output_dir=output_dir
                )
                results[dataset_file.name] = dataset_results
                
            except Exception as e:
                logger.error(f"Validation failed for {dataset_file}: {str(e)}")
                results[dataset_file.name] = {
                    'error': str(e),
                    'overall_passed': False
                }
        
        # Generate summary report
        if output_dir:
            self._generate_batch_summary(results, output_dir)
        
        return results
    
    def _generate_batch_summary(self, results: Dict[str, Dict[str, Any]], 
                              output_dir: Path):
        """Generate summary report for batch validation"""
        
        summary_lines = [
            "# Batch Validation Summary",
            f"Generated: {pd.Timestamp.now().isoformat()}",
            f"Total Datasets: {len(results)}",
            ""
        ]
        
        passed_count = sum(1 for r in results.values() 
                          if r.get('overall_passed', False))
        
        summary_lines.extend([
            f"## Summary Statistics",
            f"- Passed: {passed_count}",
            f"- Failed: {len(results) - passed_count}",
            f"- Success Rate: {passed_count / len(results) * 100:.1f}%",
            ""
        ])
        
        # Individual results
        summary_lines.append("## Individual Results")
        for dataset_name, result in results.items():
            status = "✅" if result.get('overall_passed', False) else "❌"
            total_samples = result.get('total_samples', 'N/A')
            summary_lines.append(f"- {dataset_name}: {status} ({total_samples} samples)")
        
        summary_report = "\n".join(summary_lines)
        
        summary_path = output_dir / "batch_validation_summary.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        logger.info(f"Batch validation summary saved to {summary_path}")


# CLI interface
if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    from src.pipeline_setup import Config, MetadataDB, setup_logging
    
    parser = argparse.ArgumentParser(description="Run validation pipeline")
    parser.add_argument("--synthetic-data", required=True,
                       help="Path to synthetic dataset file")
    parser.add_argument("--real-data",
                       help="Path to real dataset file (for comparison)")
    parser.add_argument("--output-dir",
                       help="Output directory for reports")
    parser.add_argument("--batch", action="store_true",
                       help="Batch validate all datasets in directory")
    
    args = parser.parse_args()
    
    # Initialize components
    config = Config()
    logger = setup_logging(config.logs_dir)
    metadata_db = MetadataDB(config.data_dir / "pipeline.db")
    
    # Create validation pipeline
    validation_pipeline = ValidationPipeline(config.model_configs, metadata_db)
    
    try:
        output_dir = Path(args.output_dir) if args.output_dir else None
        
        if args.batch:
            # Batch validation
            datasets_dir = Path(args.synthetic_data)
            real_data_path = Path(args.real_data) if args.real_data else None
            
            results = validation_pipeline.batch_validate_datasets(
                datasets_dir=datasets_dir,
                real_dataset_path=real_data_path,
                output_dir=output_dir
            )
            
            passed_count = sum(1 for r in results.values() 
                             if r.get('overall_passed', False))
            print(f"\n✅ Batch validation completed!")
            print(f"Results: {passed_count}/{len(results)} datasets passed")
        
        else:
            # Single dataset validation
            synthetic_path = Path(args.synthetic_data)
            real_path = Path(args.real_data) if args.real_data else None
            
            results = validation_pipeline.validate_synthetic_dataset(
                synthetic_dataset_path=synthetic_path,
                real_dataset_path=real_path,
                output_dir=output_dir
            )
            
            status = "✅ PASSED" if results['overall_passed'] else "❌ FAILED"
            print(f"\n{status}")
            print(f"Dataset: {results['dataset_name']}")
            print(f"Samples: {results['total_samples']}")
            
            if results['failed_checks']:
                print(f"Failed checks: {', '.join(results['failed_checks'])}")
        
    except Exception as e:
        logger.error(f"Validation pipeline failed: {e}")
        sys.exit(1)