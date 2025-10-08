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
    DataCollatorForLanguageModeling, EarlyStoppingCallback, BitsAndBytesConfig
)
from datasets import Dataset
from peft import (
    get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict,
    PeftModel, PeftConfig, prepare_model_for_kbit_training
)
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

from typing import Dict, List, Optional
from dataclasses import dataclass
from src.ingestion import UnifiedItem, LABEL_MAPS, sample_seeds_per_class

from torch.utils.data import Dataset



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

SYSTEM_ZERO_AG = (
    "Now you are a journalist writing news articles. You are given a topic and must write a "
    "corresponding news article for it. You are also given a length requirement. You must "
    "ensure your news meets the length requirement.\n\n"
    "Can you write a news report with the topic {label_name}? The length requirement is: "
    "50 words. Please be creative and write unique news articles."
)

SYSTEM_ZERO_IMDB = (
    "Now you are a movie critic. You need to have delicate emotions, unique perspectives, "
    "and a distinctive style. You are going to write a highly polar review for a movie and "
    "post it on IMDB. You are given a movie genre/style and a length requirement. You must "
    "come up with a movie that corresponds to the genre/style and write a review that meets "
    "the length requirement.\n\n"
    "Write a film review for a drama movie to express {label_name} feedback. Each review "
    "should have 80 words. Be sure to express your personal insights and feelings. "
    "Please be creative and write unique movie reviews."
)

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


# ------------ small helper dataset ------------
class _SimpleCausalDataset(Dataset):
    """
    Wraps list[str] into tokenized samples for causal LM (labels == input_ids with pad masked to -100).
    """
    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, max_len: int = 256):
        self.enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        # labels = input_ids but ignore pad tokens
        self.labels = self.enc["input_ids"].clone()
        self.labels[self.enc["attention_mask"] == 0] = -100

    def __len__(self):
        return self.enc["input_ids"].size(0)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


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
            token=True,
            torch_dtype=getattr(torch, "bfloat16", None) or torch.float16,
            device_map="auto"#self.args.device
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
            device_map="auto" #self.args.device
        )

    def _pipeline(self):
        if self._model is None or self._tok is None:
            return self._load_model()
        from transformers import pipeline as tp
        return tp("text-generation", model=self._model, tokenizer=self._tok, device_map="auto") #self.args.device)

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

def _maybe_set_bnb_version_for_cuda():
    import os, torch
    if os.getenv("BNB_CUDA_VERSION"):
        return
    cuda = (torch.version.cuda or "").strip()
    # bnb wheels commonly exist for 12.1/12.4; pick 121 if you're on a newer 12.x like 12.8
    if cuda.startswith("12") and cuda not in ("12.1", "12.4"):
        os.environ["BNB_CUDA_VERSION"] = "121"


class SyntheticTextGenerator:
    """Handles text generation using fine-tuned language models with LoRA"""
    
    def __init__(self, config: Dict[str, Any], metadata_db, models_dir: Path):
        self.config = config
        # read token from env (set by your run_serve.sh/.env)
        self.hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
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


    def load_base_model(
        self,
        model_name: Optional[str] = None,
        use_4bit: bool = True,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        """
        Loads a base Causal LM with optional 4-bit quantization and shards across all visible GPUs.

        Args:
            model_name: HF model id (falls back to self.configs["base_model_name"] if None)
            use_4bit: enable bitsandbytes 4-bit loading (saves a lot of VRAM)
            torch_dtype: fp16/bf16 for weights that remain in higher precision
        """
        name = model_name or getattr(self, "current_model_name", None) \
            or (self.configs.get("base_model_name") if hasattr(self, "configs") else None)
        if name is None:
            raise ValueError("Base model name is not set. Pass model_name or set configs['base_model_name'].")

        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        if self.tokenizer.pad_token is None:
            # for causal LMs, pad to eos
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        quant_cfg = None
        # quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        # if use_4bit:
        #     quant_cfg = BitsAndBytesConfig(
        #         load_in_4bit=True,
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_use_double_quant=True,
        #         bnb_4bit_compute_dtype=torch_dtype,
        #     )

        # Give each GPU a budget (adjust to your cards; you said ~11.8GiB)
        # max_mem = {i: "10GiB" for i in range(torch.cuda.device_count())} or {"cpu": "48GiB"}
        max_mem = {0: "8GiB", 1: "11GiB", 2: "11GiB", 3: "11GiB"}
        if torch.cuda.device_count() == 4:
            pass
        else:
            max_mem = {i: "10GiB" for i in range(torch.cuda.device_count())} 


        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            max_memory=max_mem,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            # quantization_config=quant_cfg,
            trust_remote_code=True,  # in case the model has custom code
        )
        self.model.eval()
        self.current_model_name = name

    
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
    
    def fine_tune_with_lora(
        self,
        train_texts: List[str],
        val_texts: Optional[List[str]] = None,
        experiment_name: str = "qlora_experiment",
        base_model_name: Optional[str] = None,
        # LoRA/QLoRA params
        use_qlora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        # training params
        max_seq_len: int = 256,
        learning_rate: float = 2e-4,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        warmup_ratio: float = 0.03,
        logging_steps: int = 20,
        save_steps: int = 200,
        eval_steps: Optional[int] = None,
        bf16: bool = True,
        fp16: bool = False,
        seed: int = 42,
    ) -> str:
        """
        Fine-tunes the base model with LoRA. If use_qlora=True, loads the base model in 4-bit
        and applies PEFT on top (QLoRA). Saves only the adapter weights to disk.

        Returns:
            adapter_dir (str): path to the saved LoRA/QLoRA adapter folder.
        """
        torch.manual_seed(seed)

        _maybe_set_bnb_version_for_cuda()

        # 1) Load (or reuse) base model + tokenizer
        if not hasattr(self, "model") or self.model is None:
            self.load_base_model(model_name=base_model_name, use_4bit=use_qlora, torch_dtype=torch.bfloat16 if bf16 else torch.float16)

        # 2) Prepare for k-bit training (QLoRA) & attach LoRA heads
        if use_qlora:
            self.model = prepare_model_for_kbit_training(self.model)
            # enable GC + no cache for training
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

        # Default target modules suitable for LLaMA/OPT/GPT-NeoX style blocks.
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, peft_cfg)

        # 3) Datasets
        train_ds = _SimpleCausalDataset(train_texts, self.tokenizer, max_len=max_seq_len)
        eval_ds = _SimpleCausalDataset(val_texts, self.tokenizer, max_len=max_seq_len) if val_texts else None
        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        # 4) Training args
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.models_dir) / f"{experiment_name}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        args = TrainingArguments(
            output_dir=str(out_dir),
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=2,
            eval_strategy="steps" if eval_ds is not None else "no",
            eval_steps=(eval_steps or save_steps),
            report_to=[],  # or ["mlflow"]
            bf16=bf16,
            fp16=fp16 and not bf16,
            ddp_find_unused_parameters=False,
        )

        # 5) Train
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            tokenizer=self.tokenizer,
        )
        trainer.train()

        # 6) Save adapter (PEFT saves only LoRA weights, not the base model)
        self.model.save_pretrained(str(out_dir))
        self.tokenizer.save_pretrained(str(out_dir))  # helpful for downstream

        # Optional: print trainable params summary
        try:
            from peft.utils.other import get_peft_model_state_dict
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.model.parameters())
            print(f"[LoRA] Trainable params: {n_trainable:,} / {n_total:,}")
        except Exception:
            pass

        return str(out_dir)


    def load_finetuned_model(
        self,
        adapter_dir: str,
        base_model_name: Optional[str] = None,
        use_4bit: bool = True,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        """
        Loads the base model (optionally 4-bit) and attaches a saved LoRA/QLoRA adapter for inference.
        """

        self.load_base_model(model_name=base_model_name, use_4bit=use_4bit, torch_dtype=torch_dtype)

        self.model = PeftModel.from_pretrained(self.model, adapter_dir, is_trainable=False)
        self.model.eval()

    def clean_generated_text(self, text: str, prompt: str = "") -> str:
        """Extract actual generated content, removing prompt echoes and meta-text"""
        if prompt:
            text = text.replace(prompt, "").strip()
        
        artifacts = [
            "### Task:", "### Topic:", "### Example:", 
            "Now you are a journalist", "Can you write",
            "The length requirement is", "Please be creative",
            "Thank you.\n\n"  
        ]
        for artifact in artifacts:
            if artifact in text:
                text = text.split(artifact)[0].strip()
        
        if text.startswith("Thank you"):
            lines = text.split('\n')
            text = '\n'.join(lines[1:]).strip()
        
        return text 

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
        max_new_tokens = max_length or self.gen_config.get('max_length', 100)  
        temperature = temperature or self.gen_config.get('temperature', 0.8)
        top_k = top_k or self.gen_config.get('top_k', 50)
        top_p = top_p or self.gen_config.get('top_p', 0.9)
        
        model_to_use = self.peft_model if self.peft_model else self.model
        
        if model_to_use is None:
            raise ValueError("No model loaded! Call load_base_model() or load_finetuned_model() first.")
        
        logger.info(f"Generating {num_sequences} sequences with prompt: '{prompt[:50]}...'")
        
        try:
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_length = inputs['input_ids'].shape[1]
            
            logger.info(f"Prompt length: {prompt_length} tokens, generating {max_new_tokens} new tokens")
            
            # Generate
            with torch.no_grad():
                outputs = model_to_use.generate(
                    **inputs,  # Use unpacked inputs (includes attention_mask)
                    max_new_tokens=max_new_tokens,  
                    num_return_sequences=num_sequences,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode generated sequences, removing prompt
            generated_texts = []
            for output in outputs:
                # Skip the prompt tokens
                generated_part = output[prompt_length:]
                text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
                text = self.clean_generated_text(text, prompt)
                if text.strip():  # Only add non-empty
                    generated_texts.append(text.strip())
                else:
                    logger.warning("Generated empty text, skipping")
                    
            logger.info(f"Generated {len(generated_texts)} non-empty text sequences")
            return generated_texts
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    # def generate_text(self, prompt: str = "", 
    #                  num_sequences: int = None,
    #                  max_length: int = None,
    #                  temperature: float = None,
    #                  top_k: int = None,
    #                  top_p: float = None,
    #                  do_sample: bool = True,
    #                  repetition_penalty: float = 1.1) -> List[str]:
    #     """Generate synthetic text using the model"""
        
    #     # Use model parameters or defaults
    #     num_sequences = num_sequences or self.gen_config.get('num_return_sequences', 5)
    #     max_length = max_length or self.gen_config.get('max_length', 100)
    #     temperature = temperature or self.gen_config.get('temperature', 0.8)
    #     top_k = top_k or self.gen_config.get('top_k', 50)
    #     top_p = top_p or self.gen_config.get('top_p', 0.9)
        
    #     model_to_use = self.peft_model if self.peft_model else self.model
        
    #     if model_to_use is None:
    #         raise ValueError("No model loaded! Call load_base_model() or load_finetuned_model() first.")
        
    #     logger.info(f"Generating {num_sequences} sequences with prompt: '{prompt[:50]}...'")
        
    #     try:
    #         # Tokenize prompt
    #         inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
    #         # Create attention mask for better generation quality
    #         attention_mask = torch.ones_like(inputs)
            
    #         # Generate
    #         with torch.no_grad():
    #             outputs = model_to_use.generate(
    #                 inputs,
    #                 attention_mask=attention_mask,
    #                 max_length=max_length,
    #                 num_return_sequences=num_sequences,
    #                 temperature=temperature,
    #                 top_k=top_k,
    #                 top_p=top_p,
    #                 do_sample=do_sample,
    #                 repetition_penalty=repetition_penalty,
    #                 pad_token_id=self.tokenizer.pad_token_id,
    #                 eos_token_id=self.tokenizer.eos_token_id,
    #                 # Remove early_stopping as it's deprecated in newer transformers
    #                 # early_stopping=True
    #             )
            
    #         # Decode generated sequences
    #         generated_texts = []
    #         for output in outputs:
    #             text = self.tokenizer.decode(output, skip_special_tokens=True)
    #             # Remove the prompt from the generated text
    #             if prompt:
    #                 text = text[len(prompt):].strip()
    #             generated_texts.append(text)
    #         logger.info(f"Generated {len(generated_texts)} text sequences")
    #         return generated_texts
            
    #     except Exception as e:
    #         logger.error(f"Error generating text: {str(e)}")
    #         raise
    
    def generate_dataset(self, prompts: List[str] = None,
                        num_samples: int = 100,
                        samples_per_prompt: int = 5,
                        **generation_kwargs) -> List[str]:
        """Generate a complete synthetic dataset"""        
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
            
            print(f" Fine-tuning completed! Model saved to: {model_path}")
        
        elif args.mode == "generate":
            # Load model
            if args.model_path:
                generator.load_finetuned_model(args.model_path, use_4bit=True)
            else:
                generator.load_base_model(args.model_name, use_4bit=True)
                        
            # Generate synthetic dataset
            # synthetic_texts = generator.generate_dataset(num_samples=args.num_samples)
            # after loading the (base or finetuned) model:
            if hasattr(generator, "generate_dataset"):
                synthetic_texts = generator.generate_dataset(num_samples=args.num_samples, prompts=args.prompts)
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
            
            print(f" Generated {len(synthetic_texts)} synthetic samples!")
            print(f"Saved to: {saved_path}")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)