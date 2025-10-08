# src/validate.py
"""
Comprehensive validation module for synthetic text data.
Includes quality, diversity, privacy, and similarity checks with enhanced metrics.
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
from sklearn.cluster import KMeans
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
from rouge_score import rouge_scorer
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)


def vendiscore(embeddings: np.ndarray, k: int = None) -> float:
    """
    Calculate VendiScore - a diversity metric based on eigenvalues of similarity matrix
    Higher scores indicate more diversity
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        k: Number of nearest neighbors to consider (default: sqrt(n_samples))
    
    Returns:
        VendiScore value
    """
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        return 0.0
    
    if k is None:
        k = max(1, int(np.sqrt(n_samples)))
    
    # Normalize embeddings
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    
    # Compute similarity matrix
    similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
    
    # Create kernel matrix from k-NN similarities
    kernel_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        # Get k+1 most similar (including self)
        similar_indices = np.argsort(similarity_matrix[i])[::-1]
        # Take k neighbors (excluding self if it's the most similar)
        if similar_indices[0] == i and len(similar_indices) > 1:
            neighbors = similar_indices[1:k+1]
        else:
            neighbors = similar_indices[:k]
        
        for j in neighbors:
            if j < n_samples:
                kernel_matrix[i][j] = similarity_matrix[i][j]
    
    # Make symmetric
    kernel_matrix = (kernel_matrix + kernel_matrix.T) / 2
    
    # Compute eigenvalues
    try:
        eigenvals = np.linalg.eigvals(kernel_matrix)
        eigenvals = np.real(eigenvals[eigenvals > 1e-10])  # Keep only positive eigenvalues
        
        if len(eigenvals) == 0:
            return 0.0
        
        # Normalize eigenvalues to get probability distribution
        eigenvals = eigenvals / np.sum(eigenvals)
        
        # Compute VendiScore as exponential of Shannon entropy
        entropy = -np.sum(eigenvals * np.log(eigenvals + 1e-10))
        vendi_score = np.exp(entropy)
        
        return float(vendi_score)
    except:
        return 0.0


def dcscore(texts: List[str], embedder_name: str = "princeton-nlp/unsup-simcse-bert-base-uncased", temperature: float = 1.0) -> float:
    """
    DCScore: compute embeddings -> cosine similarity matrix -> softmax row-wise with temperature -> sum diagonal.
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
    """Calculate Distinct-n metric for n-gram diversity"""
    if not texts:
        return 0.0
    
    ngrams_list = []
    for text in texts:
        tokens = text.strip().split()
        if len(tokens) >= n:
            text_ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            ngrams_list.extend(text_ngrams)
    
    if not ngrams_list:
        return 0.0
    
    unique_ngrams = set(ngrams_list)
    return len(unique_ngrams) / len(ngrams_list)


def self_bleu(texts: List[str], sample: int = 100, weights: tuple = (0.25, 0.25, 0.25, 0.25)) -> float:
    """
    Calculate Self-BLEU score for diversity measurement.
    Lower scores indicate higher diversity.
    """
    if len(texts) <= 1:
        return 0.0
    
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except Exception as e:
        raise ImportError("self_bleu requires nltk. Install: pip install nltk") from e
    
    import random
    rng = random.Random(1337)
    
    # Sample texts if too many
    sample = min(sample, len(texts))
    sampled_texts = rng.sample(texts, sample) if len(texts) > sample else texts
    
    scores = []
    smoothing = SmoothingFunction().method1
    
    for i, text in enumerate(sampled_texts):
        # Use other texts as references
        references = [t.split() for j, t in enumerate(sampled_texts) if j != i]
        hypothesis = text.split()
        
        if references and hypothesis:
            try:
                # Limit number of references for efficiency
                if len(references) > 10:
                    references = rng.sample(references, 10)
                
                score = sentence_bleu(references, hypothesis, 
                                    weights=weights,
                                    smoothing_function=smoothing)
                scores.append(score)
            except:
                continue
    
    return sum(scores) / len(scores) if scores else 0.0


def rouge_l_diversity(texts: List[str], sample_size: int = 100) -> Dict[str, float]:
    """
    Calculate ROUGE-L based diversity metrics
    Returns average ROUGE-L scores between text pairs
    """
    if len(texts) < 2:
        return {'rouge_l_precision': 0.0, 'rouge_l_recall': 0.0, 'rouge_l_f1': 0.0}
    
    # Sample texts for efficiency
    if len(texts) > sample_size:
        sampled_texts = np.random.choice(texts, sample_size, replace=False)
    else:
        sampled_texts = texts
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    # Compare each text with a few others
    n_comparisons = min(10, len(sampled_texts) - 1)
    
    for i, text1 in enumerate(sampled_texts):
        # Select comparison texts
        other_indices = [j for j in range(len(sampled_texts)) if j != i]
        comparison_indices = np.random.choice(other_indices, 
                                            min(n_comparisons, len(other_indices)), 
                                            replace=False)
        
        for j in comparison_indices:
            text2 = sampled_texts[j]
            scores = scorer.score(text1, text2)
            rouge_l = scores['rougeL']
            
            precision_scores.append(rouge_l.precision)
            recall_scores.append(rouge_l.recall)
            f1_scores.append(rouge_l.fmeasure)
    
    return {
        'rouge_l_precision': np.mean(precision_scores) if precision_scores else 0.0,
        'rouge_l_recall': np.mean(recall_scores) if recall_scores else 0.0,
        'rouge_l_f1': np.mean(f1_scores) if f1_scores else 0.0
    }


def kmeans_inertia_diversity(embeddings: np.ndarray, n_clusters: int = None) -> Dict[str, float]:
    """
    Calculate K-means inertia for cluster dispersion analysis
    Higher inertia indicates more dispersed (diverse) data
    """
    if embeddings.shape[0] < 2:
        return {'kmeans_inertia': 0.0, 'inertia_per_sample': 0.0, 'n_clusters_used': 0}
    
    if n_clusters is None:
        n_clusters = min(max(2, int(np.sqrt(embeddings.shape[0]))), embeddings.shape[0])
    
    n_clusters = min(n_clusters, embeddings.shape[0])
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        
        inertia = kmeans.inertia_
        inertia_per_sample = inertia / embeddings.shape[0]
        
        return {
            'kmeans_inertia': float(inertia),
            'inertia_per_sample': float(inertia_per_sample),
            'n_clusters_used': n_clusters
        }
    except Exception as e:
        logger.warning(f"K-means clustering failed: {e}")
        return {'kmeans_inertia': 0.0, 'inertia_per_sample': 0.0, 'n_clusters_used': 0}


# def summarize_quality(texts: List[str], compute_dc: bool = False) -> Dict[str, float]:
#     """Enhanced quality summary with additional metrics"""
#     out = {
#         "distinct_2": distinct_n(texts, 2),
#         "distinct_3": distinct_n(texts, 3),
#         "distinct_4": distinct_n(texts, 4),
#     }
    
#     try:
#         out["self_bleu"] = self_bleu(texts)
#     except Exception:
#         out["self_bleu"] = float("nan")
    
#     if compute_dc:
#         try:
#             out["dcscore"] = dcscore(texts)
#         except Exception:
#             out["dcscore"] = float("nan")
    
#     # Add ROUGE-L diversity
#     try:
#         rouge_metrics = rouge_l_diversity(texts)
#         out.update(rouge_metrics)
#     except Exception:
#         out.update({'rouge_l_precision': float("nan"), 'rouge_l_recall': float("nan"), 'rouge_l_f1': float("nan")})
    
#     return out


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
        """Calculate comprehensive diversity metrics for text dataset"""
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
        
        # Enhanced n-gram diversity (Distinct-n)
        for n in range(1, 5):
            results[f'distinct_{n}'] = distinct_n(texts, n)
        
        # Self-BLEU (diversity within generated texts)
        results['self_bleu'] = self_bleu(texts)
        
        # ROUGE-L diversity
        rouge_metrics = rouge_l_diversity(texts)
        results.update(rouge_metrics)
        
        # Get embeddings for advanced metrics
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            
            # K-means inertia (cluster dispersion)
            kmeans_metrics = kmeans_inertia_diversity(embeddings)
            results.update(kmeans_metrics)
            
            # VendiScore
            vendi_score = vendiscore(embeddings)
            results['vendi_score'] = vendi_score
            
            # DCScore
            try:
                results['dc_score'] = dcscore(texts)
            except Exception as e:
                logger.warning(f"DCScore calculation failed: {e}")
                results['dc_score'] = float('nan')
            
        except Exception as e:
            logger.warning(f"Advanced diversity metrics failed: {e}")
            results.update({
                'kmeans_inertia': float('nan'),
                'inertia_per_sample': float('nan'),
                'n_clusters_used': 0,
                'vendi_score': float('nan'),
                'dc_score': float('nan')
            })
        
        logger.info(f"Diversity metrics calculated: {len(results)} metrics")
        return results
    
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
        
        # 2. Enhanced diversity metrics
        try:
            diversity_results = self.calculate_diversity_metrics(synthetic_texts)
            
            # Check diversity thresholds
            min_uniqueness = self.thresholds.get('min_uniqueness_ratio', 0.7)
            min_distinct_2 = self.thresholds.get('min_distinct_2', 0.4)
            max_self_bleu = self.thresholds.get('max_self_bleu', 0.4)
            
            diversity_passed = (
                diversity_results['uniqueness_ratio'] >= min_uniqueness and
                diversity_results.get('distinct_2', 0) >= min_distinct_2 and
                diversity_results.get('self_bleu', 1.0) <= max_self_bleu
            )
            
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
        
        logger.info(f"Validation completed. Overall passed: {validation_results['overall_passed']}")
        return validation_results
    
    def _log_validation_to_mlflow(self, results: Dict[str, Any]):
        """Enhanced MLflow logging with all metrics"""
        try:
            with mlflow.start_run(run_name=f"validation_{results['dataset_name']}"):
                # Log basic metrics
                mlflow.log_param("dataset_name", results['dataset_name'])
                mlflow.log_param("total_samples", results['total_samples'])
                mlflow.log_param("overall_passed", results['overall_passed'])
                
                # Log detailed metrics for each validation type
                for validation_type, validation_data in results['validations'].items():
                    if isinstance(validation_data, dict) and 'error' not in validation_data:
                        
                        # Log pass/fail status
                        mlflow.log_param(f"{validation_type}_passed", validation_data.get('passed', False))
                        
                        # Log numeric metrics
                        for key, value in validation_data.items():
                            if isinstance(value, (int, float)) and not np.isnan(value) and key != 'passed':
                                mlflow.log_metric(f"{validation_type}_{key}", value)
                        
                        # Special handling for diversity metrics
                        if validation_type == 'diversity':
                            # Log all distinct-n metrics
                            for n in range(1, 5):
                                distinct_key = f'distinct_{n}'
                                if distinct_key in validation_data:
                                    mlflow.log_metric(f"distinct_{n}", validation_data[distinct_key])
                            
                            # Log ROUGE-L metrics
                            rouge_metrics = ['rouge_l_precision', 'rouge_l_recall', 'rouge_l_f1']
                            for metric in rouge_metrics:
                                if metric in validation_data and not np.isnan(validation_data[metric]):
                                    mlflow.log_metric(metric, validation_data[metric])
                            
                            # Log advanced metrics
                            advanced_metrics = ['vendi_score', 'dc_score', 'kmeans_inertia', 'inertia_per_sample']
                            for metric in advanced_metrics:
                                if metric in validation_data and not np.isnan(validation_data[metric]):
                                    mlflow.log_metric(metric, validation_data[metric])
                
                # Log failed checks
                mlflow.log_param("failed_checks", ','.join(results['failed_checks']))
                
                # Log summary metrics
                if 'diversity' in results['validations']:
                    diversity_data = results['validations']['diversity']
                    if isinstance(diversity_data, dict) and 'error' not in diversity_data:
                        # Calculate diversity score (composite metric)
                        diversity_score = (
                            diversity_data.get('distinct_2', 0) * 0.3 +
                            diversity_data.get('vendi_score', 0) * 0.3 +
                            (1 - diversity_data.get('self_bleu', 0)) * 0.2 +  # Inverted since lower is better
                            diversity_data.get('uniqueness_ratio', 0) * 0.2
                        )
                        mlflow.log_metric("diversity_composite_score", diversity_score)
                
                logger.info("Validation results logged to MLflow")
                
        except Exception as e:
            logger.warning(f"Could not log to MLflow: {str(e)}")
    
    def generate_validation_report(self, results: Dict[str, Any], 
                                 output_path: Path = None) -> str:
        """Generate a human-readable validation report with enhanced metrics"""
        
        report_lines = [
            f"# Validation Report for {results['dataset_name']}",
            f"Generated: {results['timestamp']}",
            f"Total Samples: {results['total_samples']}",
            f"Overall Status: {' PASSED' if results['overall_passed'] else ' FAILED'}",
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
                status = " PASSED" if validation_data.get('passed', False) else " FAILED"
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
                        "### Basic Diversity Metrics",
                        f"- Uniqueness Ratio: {validation_data.get('uniqueness_ratio', 'N/A'):.3f}",
                        f"- Vocabulary Size: {validation_data.get('vocabulary_size', 'N/A')}",
                        f"- Average Length: {validation_data.get('avg_length', 'N/A'):.1f}",
                        "",
                        "### N-gram Diversity (Distinct-n)",
                        f"- Distinct-1: {validation_data.get('distinct_1', 'N/A'):.3f}",
                        f"- Distinct-2: {validation_data.get('distinct_2', 'N/A'):.3f}",
                        f"- Distinct-3: {validation_data.get('distinct_3', 'N/A'):.3f}",
                        f"- Distinct-4: {validation_data.get('distinct_4', 'N/A'):.3f}",
                        "",
                        "### Lexical Overlap Metrics",
                        f"- Self-BLEU Score: {validation_data.get('self_bleu', 'N/A'):.3f} (lower is better)",
                        f"- ROUGE-L Precision: {validation_data.get('rouge_l_precision', 'N/A'):.3f}",
                        f"- ROUGE-L Recall: {validation_data.get('rouge_l_recall', 'N/A'):.3f}",
                        f"- ROUGE-L F1: {validation_data.get('rouge_l_f1', 'N/A'):.3f}",
                        "",
                        "### Advanced Diversity Metrics",
                        f"- VendiScore: {validation_data.get('vendi_score', 'N/A'):.3f} (higher is better)",
                        f"- DC Score: {validation_data.get('dc_score', 'N/A'):.3f}",
                        f"- K-means Inertia: {validation_data.get('kmeans_inertia', 'N/A'):.3f}",
                        f"- Inertia per Sample: {validation_data.get('inertia_per_sample', 'N/A'):.3f}",
                        f"- Clusters Used: {validation_data.get('n_clusters_used', 'N/A')}",
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
                        f"- Real Internal Similarity: {validation_data.get('real_internal_similarity', 'N/A'):.3f}",
                        f"- Synthetic Internal Similarity: {validation_data.get('synthetic_internal_similarity', 'N/A'):.3f}",
                        f"- Distribution Similarity: {validation_data.get('distribution_similarity', 'N/A'):.3f}",
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
        """Generate enhanced summary report for batch validation"""
        
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
        
        # Aggregate metrics across all datasets
        all_diversity_metrics = []
        all_quality_metrics = []
        
        for dataset_name, result in results.items():
            if result.get('overall_passed', False) and 'validations' in result:
                diversity_data = result['validations'].get('diversity', {})
                quality_data = result['validations'].get('quality', {})
                
                if isinstance(diversity_data, dict) and 'error' not in diversity_data:
                    all_diversity_metrics.append(diversity_data)
                
                if isinstance(quality_data, dict) and 'error' not in quality_data:
                    all_quality_metrics.append(quality_data)
        
        # Calculate aggregate statistics
        if all_diversity_metrics:
            summary_lines.extend([
                "## Aggregate Diversity Metrics (Passed Datasets Only)",
                f"- Average Distinct-2: {np.mean([d.get('distinct_2', 0) for d in all_diversity_metrics]):.3f}",
                f"- Average Self-BLEU: {np.mean([d.get('self_bleu', 0) for d in all_diversity_metrics]):.3f}",
                f"- Average VendiScore: {np.mean([d.get('vendi_score', 0) for d in all_diversity_metrics if not np.isnan(d.get('vendi_score', np.nan))]):.3f}",
                f"- Average Uniqueness: {np.mean([d.get('uniqueness_ratio', 0) for d in all_diversity_metrics]):.3f}",
                ""
            ])
        
        if all_quality_metrics:
            summary_lines.extend([
                "## Aggregate Quality Metrics (Passed Datasets Only)",
                f"- Average Perplexity: {np.mean([q.get('average_perplexity', 0) for q in all_quality_metrics]):.2f}",
                ""
            ])
        
        # Individual results
        summary_lines.append("## Individual Results")
        for dataset_name, result in results.items():
            status = "✅" if result.get('overall_passed', False) else "❌"
            total_samples = result.get('total_samples', 'N/A')
            failed_checks = result.get('failed_checks', [])
            
            line = f"- {dataset_name}: {status} ({total_samples} samples)"
            if failed_checks:
                line += f" - Failed: {', '.join(failed_checks)}"
            summary_lines.append(line)
        
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
    
    parser = argparse.ArgumentParser(description="Run enhanced validation pipeline")
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
            
            # Print key metrics
            if 'diversity' in results['validations']:
                div_data = results['validations']['diversity']
                if isinstance(div_data, dict) and 'error' not in div_data:
                    print(f"\nKey Diversity Metrics:")
                    print(f"- Distinct-2: {div_data.get('distinct_2', 'N/A'):.3f}")
                    print(f"- Self-BLEU: {div_data.get('self_bleu', 'N/A'):.3f}")
                    print(f"- VendiScore: {div_data.get('vendi_score', 'N/A'):.3f}")
                    print(f"- Uniqueness: {div_data.get('uniqueness_ratio', 'N/A'):.3f}")
        
    except Exception as e:
        logger.error(f"Validation pipeline failed: {e}")
        sys.exit(1)