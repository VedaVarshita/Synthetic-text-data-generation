# src/generator.py
"""
Text generation module using Hugging Face Transformers with PEFT/LoRA
for efficient fine-tuning and synthetic data generation.
"""

import os
import json
import logging
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from datasets import Dataset
from peft import (
    get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict,
    PeftModel, PeftConfig
)
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

from typing import Dict, List, Optional
from dataclasses import dataclass
from src.ingestion import UnifiedItem, LABEL_MAPS, sample_seeds_per_class

# ---- Lazy imports (avoid hard deps when importing the module) ----
def _lazy_import_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
    except Exception as e:
        raise ImportError("Requires 'transformers' and 'torch'. Install: pip install transformers torch") from e
    return torch, AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig

def _lazy_import_peft():
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise ImportError("LoRA mode needs 'peft'. Install: pip install peft") from e
    return LoraConfig, get_peft_model

# ---------------------- Prompt templates ----------------------
SYSTEM_ZERO_IMDB = """You are an IMDB movie reviewer. Write a concise {label_name} review.
- Keep it natural and varied.
- Avoid explicit content.
Length: 60-120 words."""
SYSTEM_ZERO_AG = """You are a news editor. Write a concise news summary for the category: {label_name}.
- Keep it factual and varied.
- Include plausible named entities.
Length: 40-80 words."""
FEWSHOT_HEADER = "Here are a few examples:\n\n"
FEWSHOT_LINE = "\"{text}\" ({label_name})\n"
INSTRUCTION_ZERO = "\nNow write a new example:"

def build_prompt(dataset: str, label_id: int, label_name: str, mode: str, seed_examples: Optional[List[UnifiedItem]] = None) -> str:
    dataset = dataset.lower()
    if mode == "zero_shot":
        base = SYSTEM_ZERO_IMDB if dataset == "imdb" else SYSTEM_ZERO_AG
        return base.format(label_name=label_name) + INSTRUCTION_ZERO

    if mode == "few_shot":
        assert seed_examples is not None and len(seed_examples) > 0
        label_map = LABEL_MAPS[dataset]
        base = (SYSTEM_ZERO_IMDB if dataset == "imdb" else SYSTEM_ZERO_AG).format(label_name=label_name)
        lines = []
        for ex in seed_examples:
            lines.append(FEWSHOT_LINE.format(
                text=ex.text.replace("\n", " ").strip(),
                label_name=label_map[ex.label]
            ))
        return base + "\n\n" + FEWSHOT_HEADER + "".join(lines) + INSTRUCTION_ZERO

    raise ValueError(f"Unsupported mode: {mode}")

# ---------------------- Generator ----------------------
@dataclass
class GenArgs:
    model_name: str
    device: str = "auto"
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True
    batch_size: int = 4
    seed: int = 42
    lora: Optional[dict] = None  # LoRAConfig dict-like if provided

class SyntheticGenerator:
    def __init__(self, dataset: str, label_map: Dict[int, str], gen_args: GenArgs):
        self.dataset = dataset
        self.label_map = label_map
        self.args = gen_args
        self._torch = None
        self._model = None
        self._tok = None

    def _load_model(self):
        torch, AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig = _lazy_import_transformers()
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype=getattr(torch, "bfloat16", None) or torch.float16,
            device_map=self.args.device
        )
        # If LoRA requested, wrap the model (you can train adapters before generation if desired)
        if self.args.lora:
            LoraConfig, get_peft_model = _lazy_import_peft()
            cfg = self.args.lora
            target_modules = cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]
            peft_cfg = LoraConfig(
                r=cfg.get("r", 16),
                lora_alpha=cfg.get("alpha", 32),
                lora_dropout=cfg.get("dropout", 0.05),
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self._model = get_peft_model(self._model, peft_cfg)

        return pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tok,
            device_map=self.args.device
        )

    def _pipeline(self):
        if self._model is None or self._tok is None:
            return self._load_model()
        from transformers import pipeline as tp
        return tp("text-generation", model=self._model, tokenizer=self._tok, device_map=self.args.device)

    def _generate(self, prompts: List[str]) -> List[str]:
        pipe = self._pipeline()
        outputs = pipe(
            prompts,
            max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            do_sample=self.args.do_sample,
            pad_token_id=self._tok.eos_token_id,
            batch_size=self.args.batch_size
        )
        texts = []
        for out in outputs:
            if isinstance(out, list):
                out = out[0]
            texts.append(out["generated_text"])
        return texts

    def synthesize_class(self, label_id: int, n: int, mode: str, seed_examples: Optional[List[UnifiedItem]] = None) -> List[str]:
        label_name = self.label_map[label_id]
        prompts = [build_prompt(self.dataset, label_id, label_name, mode, seed_examples) for _ in range(n)]
        return self._generate(prompts)

    def fewshot_seeds(self, train_items: List[UnifiedItem], k_per_class: int, num_classes: int):
        return sample_seeds_per_class(train_items, k_per_class, num_classes)


class SyntheticTextGenerator:
    """Handles text generation using fine-tuned language models with LoRA"""
    
    def __init__(self, config: Dict[str, Any], metadata_db, models_dir: Path):
        self.config = config
        self.metadata_db = metadata_db
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize model components
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Generation parameters from config
        self.gen_config = config.get('generator', {})
    
    def load_base_model(self, model_name: str = None):
        """Load the base language model and tokenizer"""
        if model_name is None:
            model_name = self.gen_config.get('model_name', 'distilgpt2')
        
        logger.info(f"Loading base model: {model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            
            self.model.to(self.device)
            logger.info(f"Successfully loaded {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def setup_peft_config(self, custom_config: Dict = None) -> LoraConfig:
        """Setup PEFT LoRA configuration"""
        lora_params = custom_config or self.config.get('lora_config', {})
        
        # Default LoRA configuration for language models
        default_config = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["c_attn", "c_proj"],  # For GPT-style models
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM
        }
        
        # Update with custom parameters
        default_config.update(lora_params)
        
        logger.info(f"LoRA config: {default_config}")
        return LoraConfig(**default_config)
    
    def prepare_dataset_for_training(self, texts: List[str], 
                                   max_length: int = 256) -> Dataset:
        """Prepare text dataset for training"""
        logger.info(f"Preparing dataset with {len(texts)} texts...")
        
        # Tokenize texts
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None
            )
        
        # Create dataset
        dataset = Dataset.from_dict({'text': texts})
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        logger.info(f"Dataset prepared with {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def fine_tune_with_lora(self, train_texts: List[str], 
                          val_texts: List[str] = None,
                          experiment_name: str = "lora_finetune",
                          max_length: int = 256) -> str:
        """Fine-tune model using LoRA"""
        logger.info("Starting LoRA fine-tuning...")
        
        if self.model is None:
            self.load_base_model()
        
        # Setup LoRA
        peft_config = self.setup_peft_config()
        self.peft_model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
        # Prepare datasets
        train_dataset = self.prepare_dataset_for_training(train_texts, max_length)
        eval_dataset = None
        if val_texts:
            eval_dataset = self.prepare_dataset_for_training(val_texts, max_length)
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Not masked language modeling
            pad_to_multiple_of=8 if self.device.type == "cuda" else None
        )
        
        # Training arguments
        training_config = self.config.get('training_config', {})
        output_dir = self.models_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Base training arguments
        training_args_dict = {
            "output_dir": str(output_dir),
            "overwrite_output_dir": True,
            "num_train_epochs": training_config.get('num_train_epochs', 3),
            "per_device_train_batch_size": training_config.get('per_device_train_batch_size', 4),
            "per_device_eval_batch_size": training_config.get('per_device_eval_batch_size', 4),
            "warmup_steps": training_config.get('warmup_steps', 100),
            "weight_decay": training_config.get('weight_decay', 0.01),
            "logging_dir": str(self.models_dir.parent / "logs" / "training"),
            "logging_steps": training_config.get('logging_steps', 10),
            "save_steps": training_config.get('save_steps', 500),
            "save_total_limit": training_config.get('save_total_limit', 2),
            "report_to": None,
            "push_to_hub": False,
            "fp16": self.device.type == "cuda",
            "dataloader_pin_memory": False,
            "remove_unused_columns": False
        }

        # Add evaluation-specific arguments only if we have eval dataset
        # FIXED: Use 'eval_strategy' instead of 'evaluation_strategy' for transformers >= 4.21
        if eval_dataset:
            training_args_dict.update({
                "eval_strategy": "steps",  # Changed from 'evaluation_strategy'
                "eval_steps": training_config.get('eval_steps', 500),
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            })
        else:
            training_args_dict.update({
                "eval_strategy": "no",  # Changed from 'evaluation_strategy'
                "load_best_model_at_end": False,
            })

        training_args = TrainingArguments(**training_args_dict)
        
        # Setup trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if eval_dataset else None
        )
        
        # Start MLflow run for training
        with mlflow.start_run(run_name=f"finetune_{experiment_name}"):
            # Log parameters
            mlflow.log_params({
                "model_name": self.gen_config.get('model_name', 'distilgpt2'),
                "train_samples": len(train_texts),
                "val_samples": len(val_texts) if val_texts else 0,
                "max_length": max_length,
                **training_config,
                **peft_config.to_dict()
            })
            
            # Train the model
            logger.info("Starting training...")
            trainer.train()
            
            # Save the model
            trainer.save_model()
            
            # Log model to MLflow
            mlflow.pytorch.log_model(
                self.peft_model,
                "model",
                registered_model_name=f"synthetic_generator_{experiment_name}"
            )
            
            # Get final metrics
            if eval_dataset:
                eval_results = trainer.evaluate()
                mlflow.log_metrics(eval_results)
            
            mlflow_run_id = mlflow.active_run().info.run_id
            
            # Log experiment to database
            # Convert any sets to lists for JSON serialization
            peft_dict = peft_config.to_dict()
            json_serializable_params = {}
            
            # Combine training config and peft config
            all_params = {**training_config, **peft_dict}
            
            # Convert sets to lists for JSON serialization
            for key, value in all_params.items():
                if isinstance(value, set):
                    json_serializable_params[key] = list(value)
                else:
                    json_serializable_params[key] = value
            
            self.metadata_db.log_experiment(
                name=experiment_name,
                model_name=self.gen_config.get('model_name', 'distilgpt2'),
                parameters=json_serializable_params,
                mlflow_run_id=mlflow_run_id
            )
            
            logger.info(f"Fine-tuning completed! Model saved to {output_dir}")
            return str(output_dir)
    
    def load_finetuned_model(self, model_path: str):
        """Load a fine-tuned LoRA model"""
        logger.info(f"Loading fine-tuned model from {model_path}")
        
        try:
            # Load base model if not already loaded
            if self.model is None:
                self.load_base_model()
            
            # Load PEFT model
            self.peft_model = PeftModel.from_pretrained(self.model, model_path)
            logger.info("Fine-tuned model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {str(e)}")
            raise
    
    def generate_text(self, prompt: str = "", 
                     num_sequences: int = None,
                     max_length: int = None,
                     temperature: float = None,
                     top_k: int = None,
                     top_p: float = None,
                     do_sample: bool = True,
                     repetition_penalty: float = 1.1) -> List[str]:
        """Generate synthetic text using the model"""
        
        # Use model parameters or defaults
        num_sequences = num_sequences or self.gen_config.get('num_return_sequences', 5)
        max_length = max_length or self.gen_config.get('max_length', 100)
        temperature = temperature or self.gen_config.get('temperature', 0.8)
        top_k = top_k or self.gen_config.get('top_k', 50)
        top_p = top_p or self.gen_config.get('top_p', 0.9)
        
        model_to_use = self.peft_model if self.peft_model else self.model
        
        if model_to_use is None:
            raise ValueError("No model loaded! Call load_base_model() or load_finetuned_model() first.")
        
        logger.info(f"Generating {num_sequences} sequences with prompt: '{prompt[:50]}...'")
        
        try:
            # Tokenize prompt
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Create attention mask for better generation quality
            attention_mask = torch.ones_like(inputs)
            
            # Generate
            with torch.no_grad():
                outputs = model_to_use.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_return_sequences=num_sequences,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # Remove early_stopping as it's deprecated in newer transformers
                    # early_stopping=True
                )
            
            # Decode generated sequences
            generated_texts = []
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Remove the prompt from the generated text
                if prompt:
                    text = text[len(prompt):].strip()
                generated_texts.append(text)
            
            logger.info(f"Generated {len(generated_texts)} text sequences")
            return generated_texts
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    def generate_dataset(self, prompts: List[str] = None,
                        num_samples: int = 100,
                        samples_per_prompt: int = 5,
                        **generation_kwargs) -> List[str]:
        """Generate a complete synthetic dataset"""
        
        if prompts is None:
            # Use some default prompts for general text generation
            prompts = [
                "The weather today is",
                "In the future, technology will",
                "One interesting fact about",
                "The best way to learn",
                "An important lesson in life is",
                "When traveling to new places",
                "The key to success is",
                "In order to solve problems",
                "The most important thing to remember",
                "A good strategy for"
            ]
        
        logger.info(f"Generating dataset with {num_samples} samples using {len(prompts)} prompts")
        
        all_generated = []
        samples_needed = num_samples
        
        # Generate in batches using different prompts
        while len(all_generated) < num_samples and samples_needed > 0:
            for prompt in prompts:
                if len(all_generated) >= num_samples:
                    break
                
                batch_size = min(samples_per_prompt, samples_needed)
                
                generated_batch = self.generate_text(
                    prompt=prompt,
                    num_sequences=batch_size,
                    **generation_kwargs
                )
                
                # Filter out very short or empty generations
                valid_generations = [text for text in generated_batch 
                                   if text.strip() and len(text.strip()) > 20]
                
                all_generated.extend(valid_generations)
                samples_needed = num_samples - len(all_generated)
        
        # Trim to exact number if we generated too many
        final_dataset = all_generated[:num_samples]
        
        logger.info(f"Generated synthetic dataset with {len(final_dataset)} samples")
        return final_dataset
    
    def save_synthetic_dataset(self, generated_texts: List[str],
                              dataset_name: str,
                              version: str = "v1",
                              metadata: Dict = None) -> Path:
        """Save generated dataset and log to database"""
        
        synthetic_dir = self.models_dir.parent / "data" / "synthetic"
        synthetic_dir.mkdir(exist_ok=True)
        
        filename = f"{dataset_name}_{version}_synthetic.json"
        file_path = synthetic_dir / filename
        
        # Create dataset structure
        dataset_info = {
            'dataset_name': dataset_name,
            'version': version,
            'type': 'synthetic',
            'size': len(generated_texts),
            'created_at': datetime.now().isoformat(),
            'generator_config': self.gen_config,
            'metadata': metadata or {},
            'texts': generated_texts
        }
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        # Log to database
        db_metadata = {
            'generator_model': self.gen_config.get('model_name', 'unknown'),
            'avg_length': sum(len(text) for text in generated_texts) / len(generated_texts),
            'generation_params': self.gen_config,
            **(metadata or {})
        }
        
        dataset_id = self.metadata_db.log_dataset(
            name=dataset_name,
            version=version,
            type_="synthetic",
            size=len(generated_texts),
            file_path=str(file_path),
            metadata=db_metadata
        )
        
        logger.info(f"Saved synthetic dataset to {file_path} (DB ID: {dataset_id})")
        return file_path


# Example usage and CLI interface
if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    from src.pipeline_setup import Config, MetadataDB, setup_logging
    
    parser = argparse.ArgumentParser(description="Run text generation pipeline")
    parser.add_argument("--mode", choices=["finetune", "generate"], required=True,
                       help="Mode: finetune or generate")
    parser.add_argument("--model-name", default="distilgpt2",
                       help="Base model name")
    parser.add_argument("--data-path", 
                       help="Path to training data (for finetune) or model (for generate)")
    parser.add_argument("--experiment-name", default="experiment_1",
                       help="Experiment name")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of synthetic samples to generate")
    parser.add_argument("--dataset-name", default="synthetic_text",
                       help="Name for synthetic dataset")
    parser.add_argument("--model-path", help="path to a finetuned model dir")
    
    args = parser.parse_args()
    
    # Initialize components
    config = Config()
    logger = setup_logging(config.logs_dir)
    metadata_db = MetadataDB(config.data_dir / "pipeline.db")
    
    # Create generator
    generator = SyntheticTextGenerator(config.model_configs, metadata_db, config.models_dir)
    
    try:
        if args.mode == "finetune":
            # Load training data
            if not args.data_path:
                print("Error: --data-path required for finetune mode")
                sys.exit(1)
            
            # Load data from processed files
            data_path = Path(args.data_path)
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            train_texts = data['texts']
            
            # Fine-tune model
            model_path = generator.fine_tune_with_lora(
                train_texts=train_texts,
                experiment_name=args.experiment_name
            )
            
            print(f"✅ Fine-tuning completed! Model saved to: {model_path}")
        
        elif args.mode == "generate":
            # Load model
            if args.model_path:
                generator.load_finetuned_model(args.model_path)
            else:
                generator.load_base_model(args.model_name)
                        
            # Generate synthetic dataset
            # synthetic_texts = generator.generate_dataset(num_samples=args.num_samples)
            # after loading the (base or finetuned) model:
            if hasattr(generator, "generate_dataset"):
                synthetic_texts = generator.generate_dataset(num_samples=args.num_samples)
            else:
                # fallback using generate_text loop
                prompts = args.prompts or [""]           # if you support passing prompts
                num = args.num_samples
                synthetic_texts = []
                i = 0
                while len(synthetic_texts) < num:
                    p = prompts[i % len(prompts)]
                    take = min(args.num_return_sequences or 1, num - len(synthetic_texts))
                    seqs = generator.generate_text(
                        prompt=p,
                        num_sequences=take,
                        max_length=args.max_length,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                    )
                    synthetic_texts.extend(seqs)
                    i += 1

            
            # Save dataset
            saved_path = generator.save_synthetic_dataset(
                synthetic_texts,
                args.dataset_name,
                metadata={'experiment': args.experiment_name}
            )
            
            print(f"✅ Generated {len(synthetic_texts)} synthetic samples!")
            print(f"Saved to: {saved_path}")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)