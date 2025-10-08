# src/serve.py
"""
FastAPI service for the synthetic data generation pipeline.
Provides REST endpoints for generating, validating, and managing synthetic datasets.
"""


import os
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Synthetic Text Generation API",
    description="API for generating and validating synthetic text datasets",
    version="1.0.0"
)

# Global variables for models and pipeline components
generator = None
validator = None
config = None
metadata_db = None

# Pydantic models for request/response
class GenerationRequest(BaseModel):
    prompts: Optional[List[str]] = None
    num_samples: int = Field(default=100, ge=1, le=10000)
    samples_per_prompt: int = Field(default=5, ge=1, le=50)
    max_length: Optional[int] = Field(default=100, ge=10, le=500)
    temperature: Optional[float] = Field(default=0.8, ge=0.1, le=2.0)
    top_k: Optional[int] = Field(default=50, ge=1, le=100)
    top_p: Optional[float] = Field(default=0.9, ge=0.1, le=1.0)
    dataset_name: str = Field(default="api_generated")
    use_finetuned: bool = Field(default=False)
    model_path: Optional[str] = None

class ValidationRequest(BaseModel):
    synthetic_dataset_id: Optional[str] = None
    synthetic_texts: Optional[List[str]] = None
    real_dataset_id: Optional[str] = None
    real_texts: Optional[List[str]] = None
    dataset_name: str = Field(default="validation_dataset")

class FineTuneRequest(BaseModel):
    training_texts: List[str]
    validation_texts: Optional[List[str]] = None
    experiment_name: str = Field(default="api_finetune")
    num_epochs: int = Field(default=3, ge=1, le=10)
    batch_size: int = Field(default=4, ge=1, le=16)
    learning_rate: float = Field(default=5e-5, ge=1e-6, le=1e-3)

class GenerationResponse(BaseModel):
    dataset_id: str
    dataset_name: str
    generated_texts: List[str]
    generation_metadata: Dict[str, Any]
    file_path: Optional[str] = None

class ValidationResponse(BaseModel):
    validation_id: str
    dataset_name: str
    overall_passed: bool
    failed_checks: List[str]
    detailed_results: Dict[str, Any]
    report_path: Optional[str] = None

class DatasetInfo(BaseModel):
    id: int
    name: str
    version: str
    type: str
    size: int
    created_at: str
    file_path: str
    metadata: Dict[str, Any]

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize pipeline components on startup"""
    global generator, validator, config, metadata_db
    
    logger.info("Initializing API components...")
    
    try:
        # Setup project paths
        project_root = Path(__file__).parent.parent
        
        # Import pipeline components
        import sys
        sys.path.append(str(project_root))
        
        from src.pipeline_setup import Config, MetadataDB
        from src.generator import SyntheticTextGenerator
        from src.validate import TextValidator
        
        # Initialize configuration
        config = Config()
        
        # Initialize database
        metadata_db = MetadataDB(config.data_dir / "pipeline.db")
        
        # Initialize generator (load base model)
        generator = SyntheticTextGenerator(config.model_configs, metadata_db, config.models_dir)
        generator.load_base_model()
        
        # Initialize validator
        validator = TextValidator(config.model_configs, metadata_db)
        
        logger.info("API components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize API components: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "Welcome to the Synthetic Text Generation API. See /docs for API reference."}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "generator": generator is not None,
            "validator": validator is not None,
            "database": metadata_db is not None
        }
    }

# Generation endpoints
@app.post("/generate", response_model=GenerationResponse)
async def generate_synthetic_data(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate synthetic text dataset"""
    
    if generator is None:
        raise HTTPException(status_code=500, detail="Generator not initialized")
    
    try:
        # Load fine-tuned model if requested
        if request.use_finetuned and request.model_path:
            generator.load_finetuned_model(request.model_path)
        elif request.use_finetuned and not request.model_path:
            raise HTTPException(status_code=400, 
                              detail="model_path required when use_finetuned=True")
        
        # Generate synthetic dataset
        generation_kwargs = {}
        if request.max_length: generation_kwargs['max_length'] = request.max_length
        if request.temperature: generation_kwargs['temperature'] = request.temperature
        if request.top_k: generation_kwargs['top_k'] = request.top_k
        if request.top_p: generation_kwargs['top_p'] = request.top_p
        
        synthetic_texts = generator.generate_dataset(
            prompts=request.prompts,
            num_samples=request.num_samples,
            samples_per_prompt=request.samples_per_prompt,
            **generation_kwargs
        )
        
        # Save dataset
        metadata = {
            "api_request": request.dict(),
            "generation_timestamp": datetime.now().isoformat()
        }
        
        saved_path = generator.save_synthetic_dataset(
            synthetic_texts,
            request.dataset_name,
            metadata=metadata
        )
        
        # Generate response
        dataset_id = f"{request.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        response = GenerationResponse(
            dataset_id=dataset_id,
            dataset_name=request.dataset_name,
            generated_texts=synthetic_texts,
            generation_metadata=metadata,
            file_path=str(saved_path)
        )
        
        logger.info(f"Generated {len(synthetic_texts)} synthetic samples for {request.dataset_name}")
        return response
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Fine-tuning endpoint
@app.post("/finetune")
async def finetune_model(request: FineTuneRequest, background_tasks: BackgroundTasks):
    """Fine-tune model with custom data"""
    
    if generator is None:
        raise HTTPException(status_code=500, detail="Generator not initialized")
    
    try:
        # Start fine-tuning in background
        def run_finetune():
            model_path = generator.fine_tune_with_lora(
                train_texts=request.training_texts,
                val_texts=request.validation_texts,
                experiment_name=request.experiment_name
            )
            logger.info(f"Fine-tuning completed: {model_path}")
            return model_path
        
        # Add to background tasks
        background_tasks.add_task(run_finetune)
        
        return {
            "status": "started",
            "experiment_name": request.experiment_name,
            "message": "Fine-tuning started in background",
            "training_samples": len(request.training_texts),
            "validation_samples": len(request.validation_texts) if request.validation_texts else 0
        }
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Validation endpoints
@app.post("/validate", response_model=ValidationResponse)
async def validate_dataset(request: ValidationRequest):
    """Validate synthetic dataset quality"""
    
    if validator is None:
        raise HTTPException(status_code=500, detail="Validator not initialized")
    
    try:
        # Get synthetic texts
        synthetic_texts = []
        if request.synthetic_texts:
            synthetic_texts = request.synthetic_texts
        elif request.synthetic_dataset_id:
            # Load from database or file
            # This is a simplified version - in practice you'd query the DB
            raise HTTPException(status_code=501, detail="Dataset loading not implemented yet")
        else:
            raise HTTPException(status_code=400, 
                              detail="Either synthetic_texts or synthetic_dataset_id required")
        
        # Get real texts for comparison
        real_texts = None
        if request.real_texts:
            real_texts = request.real_texts
        elif request.real_dataset_id:
            # Load from database or file
            pass
        
        # Run validation
        validation_results = validator.validate_dataset(
            synthetic_texts=synthetic_texts,
            real_texts=real_texts,
            dataset_name=request.dataset_name
        )
        
        # Generate validation ID
        validation_id = f"val_{request.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        response = ValidationResponse(
            validation_id=validation_id,
            dataset_name=request.dataset_name,
            overall_passed=validation_results['overall_passed'],
            failed_checks=validation_results['failed_checks'],
            detailed_results=validation_results['validations']
        )
        
        logger.info(f"Validation completed for {request.dataset_name}: "
                   f"{'PASSED' if validation_results['overall_passed'] else 'FAILED'}")
        
        return response
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Dataset management endpoints
@app.get("/datasets", response_model=List[DatasetInfo])
async def list_datasets(dataset_type: Optional[str] = None):
    """List all datasets in the database"""
    
    if metadata_db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        # Query database for datasets
        import sqlite3
        
        query = "SELECT * FROM datasets"
        params = []
        
        if dataset_type:
            query += " WHERE type = ?"
            params.append(dataset_type)
        
        query += " ORDER BY created_at DESC"
        
        with sqlite3.connect(metadata_db.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        # Convert to response format
        datasets = []
        for row in rows:
            metadata_dict = json.loads(row['metadata']) if row['metadata'] else {}
            dataset_info = DatasetInfo(
                id=row['id'],
                name=row['name'],
                version=row['version'],
                type=row['type'],
                size=row['size'],
                created_at=row['created_at'],
                file_path=row['file_path'],
                metadata=metadata_dict
            )
            datasets.append(dataset_info)
        
        return datasets
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: int):
    """Get specific dataset details"""
    
    if metadata_db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        import sqlite3
        
        with sqlite3.connect(metadata_db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load dataset file if it exists
        file_path = Path(row['file_path'])
        dataset_content = None
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                dataset_content = json.load(f)
        
        metadata_dict = json.loads(row['metadata']) if row['metadata'] else {}
        
        return {
            "id": row['id'],
            "name": row['name'],
            "version": row['version'],
            "type": row['type'],
            "size": row['size'],
            "created_at": row['created_at'],
            "file_path": row['file_path'],
            "metadata": metadata_dict,
            "content": dataset_content
        }
        
    except Exception as e:
        logger.error(f"Failed to get dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: int):
    """Download dataset file"""
    
    if metadata_db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        import sqlite3
        
        with sqlite3.connect(metadata_db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path, name FROM datasets WHERE id = ?", (dataset_id,))
            row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        file_path = Path(row[0])
        dataset_name = row[1]
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
        
        return FileResponse(
            path=str(file_path),
            filename=f"{dataset_name}.json",
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Failed to download dataset {dataset_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# File upload endpoint
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...), dataset_type: str = "real"):
    """Upload a dataset file"""
    
    if metadata_db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        # Validate file type
        if not file.filename.endswith(('.json', '.txt', '.jsonl')):
            raise HTTPException(status_code=400, 
                              detail="Unsupported file type. Use .json, .txt, or .jsonl")
        
        # Save uploaded file
        upload_dir = config.data_dir / "uploads"
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Parse file content
        texts = []
        if file.filename.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts = [str(item) for item in data]
                elif isinstance(data, dict) and 'texts' in data:
                    texts = data['texts']
        elif file.filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                texts = content.strip().split('\n')
        elif file.filename.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if isinstance(data, str):
                        texts.append(data)
                    elif isinstance(data, dict) and 'text' in data:
                        texts.append(data['text'])
        
        # Save to database
        dataset_name = file.filename.split('.')[0]
        metadata = {
            "upload_timestamp": datetime.now().isoformat(),
            "original_filename": file.filename,
            "file_size": len(content)
        }
        
        dataset_id = metadata_db.save_dataset(
            name=dataset_name,
            version="1.0",
            dataset_type=dataset_type,
            file_path=str(file_path),
            size=len(texts),
            metadata=metadata
        )
        
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "file_path": str(file_path),
            "num_texts": len(texts),
            "type": dataset_type,
            "message": "Dataset uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to upload dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Statistics endpoint
@app.get("/stats")
async def get_stats():
    """Get pipeline statistics"""
    
    if metadata_db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        import sqlite3
        
        with sqlite3.connect(metadata_db.db_path) as conn:
            cursor = conn.cursor()
            
            # Get dataset counts
            cursor.execute("SELECT type, COUNT(*) FROM datasets GROUP BY type")
            dataset_counts = dict(cursor.fetchall())
            
            # Get total samples
            cursor.execute("SELECT SUM(size) FROM datasets")
            total_samples = cursor.fetchone()[0] or 0
            
            # Get recent activity (last 24 hours)
            cursor.execute("""
                SELECT COUNT(*) FROM datasets 
                WHERE created_at >= datetime('now', '-1 day')
            """)
            recent_datasets = cursor.fetchone()[0]
        
        return {
            "total_datasets": sum(dataset_counts.values()),
            "dataset_counts": dataset_counts,
            "total_samples": total_samples,
            "recent_datasets": recent_datasets,
            "components_status": {
                "generator": generator is not None,
                "validator": validator is not None,
                "database": metadata_db is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Synthetic Text Generation API")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "serve:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )