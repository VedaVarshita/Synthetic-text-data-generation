#!/usr/bin/env python3
"""
Setup script for the Synthetic Language Data Generation Pipeline.
This script sets up the complete project structure and verifies all components.
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess

# def run_command(command, description=""):
#     """Run a command and return success status"""
#     print(f"🔄 {description}...")
#     try:
#         result = subprocess.run(command, shell=True, capture_output=True, text=True)
#         if result.returncode == 0:
#             print(f"✅ {description} completed successfully")
#             return True
#         else:
#             print(f"❌ {description} failed: {result.stderr}")
#             return False
#     except Exception as e:
#         print(f"❌ {description} failed: {e}")
#         return False

def create_file_with_content(file_path, content):
    """Create a file with the given content"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Created {file_path}")

def setup_project_structure():
    """Create the complete project structure"""
    print("📁 Setting up project structure...")
    
    directories = [
        "data/raw", "data/processed", "data/synthetic", "data/uploads",
        "src", "pipelines/prefect", "notebooks", "models", "mlruns", 
        "logs", "reports/validation", "monitoring/grafana/dashboards",
        "monitoring/grafana/datasources", "monitoring/prometheus",
        "config", "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "pipelines/__init__.py", 
        "pipelines/prefect/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
    
    print("✅ Project structure created!")

def create_requirements_txt():
    """Create requirements.txt with all necessary dependencies"""
    requirements_content = """# Core ML and Data Science
torch>=1.13.0
transformers>=4.25.0
datasets>=2.8.0
sentence-transformers>=2.2.2
peft>=0.6.0
accelerate>=0.16.0

# MLOps and Experiment Tracking
mlflow>=2.8.0
prefect>=2.14.0

# API and Web
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0

# Data Processing
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
nltk>=3.8.0

# Database
sqlalchemy>=2.0.0

# Utilities
python-dotenv>=1.0.0
click>=8.1.0
tqdm>=4.64.0
requests>=2.28.0
python-multipart>=0.0.6

# Development and Testing
pytest>=7.4.0
jupyter>=1.0.0
ipykernel>=6.25.0
"""
    
    create_file_with_content(Path("requirements.txt"), requirements_content)

# def create_env_file():
#     """Create .env file with default configuration"""
#     env_content = """# Project Configuration
# PROJECT_NAME=synthetic-lm-pipeline
# ENVIRONMENT=development
# DEBUG=true

# # API Configuration
# API_HOST=0.0.0.0
# API_PORT=8000
# API_WORKERS=1

# # MLflow Configuration
# MLFLOW_TRACKING_URI=file://./mlruns
# MLFLOW_EXPERIMENT_NAME=synthetic_data_generation

# # Model Configuration
# DEFAULT_MODEL_NAME=distilgpt2
# EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
# MAX_SEQUENCE_LENGTH=512

# # Generation Configuration
# DEFAULT_TEMPERATURE=0.8
# DEFAULT_TOP_K=50
# DEFAULT_TOP_P=0.9
# DEFAULT_MAX_LENGTH=100

# # Validation Thresholds
# MIN_UNIQUENESS_RATIO=0.7
# MAX_EXACT_DUPLICATES=0.05
# MIN_AVG_PERPLEXITY=5.0
# MAX_AVG_PERPLEXITY=100.0
# MIN_EMBEDDING_SIMILARITY=0.3

# # Database Configuration
# DATABASE_URL=sqlite:///./data/pipeline.db

# # Storage Configuration
# DATA_DIR=./data
# MODELS_DIR=./models
# LOGS_DIR=./logs
# """
    
#     create_file_with_content(Path(".env"), env_content)

# def create_dockerignore():
#     """Create .dockerignore file"""
#     dockerignore_content = """# Git
# .git
# .gitignore

# # Python
# __pycache__
# *.pyc
# *.pyo
# *.pyd
# .Python
# env
# pip-log.txt
# pip-delete-this-directory.txt
# .venv
# venv/

# # Local development
# .env.local
# .DS_Store
# *.log

# # IDE
# .vscode
# .idea
# *.swp
# *.swo

# # Documentation
# README.md
# docs/

# # Data (you might want to keep this out of containers)
# *.csv
# *.json
# data/
# models/
# mlruns/
# """
    
#     create_file_with_content(Path(".dockerignore"), dockerignore_content)

# def create_readme():
#     """Create README.md file"""
#     readme_content = """# Synthetic Language Data Generation Pipeline

# A comprehensive end-to-end pipeline for generating and validating synthetic text data using modern MLOps practices.

# ## Features

# - 🚀 **Data Ingestion**: Load and preprocess data from various sources
# - 🧠 **Efficient Fine-tuning**: Use PEFT/LoRA for resource-efficient model training
# - 📝 **Text Generation**: Generate high-quality synthetic text with various strategies
# - ✅ **Quality Validation**: Comprehensive validation framework for synthetic data
# - 🌐 **API Service**: Production-ready FastAPI service
# - 📈 **Experiment Tracking**: MLflow integration for experiment management
# - 🔄 **Workflow Orchestration**: Prefect flows for pipeline automation
# - 🐳 **Containerized Deployment**: Docker and Docker Compose support

# ## Quick Start

# 1. **Setup the project:**
#    ```bash
#    python setup.py
#    ```

# 2. **Install dependencies:**
#    ```bash
#    pip install -r requirements.txt
#    ```

# 3. **Run the demo:**
#    ```bash
#    python demo_script.py --quick
#    ```

# 4. **Start the API:**
#    ```bash
#    python src/serve.py
#    ```

# 5. **View MLflow UI:**
#    ```bash
#    mlflow ui --backend-store-uri ./mlruns
#    ```

# ## Project Structure

# ```
# synthetic-lm-pipeline/
# ├── src/                 # Core pipeline components
# ├── pipelines/           # Workflow orchestration
# ├── data/               # Data storage
# ├── models/             # Model storage
# ├── reports/            # Validation reports
# ├── monitoring/         # Monitoring configuration
# └── notebooks/          # Jupyter notebooks
# ```

# ## Documentation

# - API Documentation: http://localhost:8000/docs (when API is running)
# - MLflow UI: http://localhost:5000 (when MLflow UI is running)

# ## License

# MIT License
# """
    
#     create_file_with_content(Path("README.md"), readme_content)

# def setup_virtual_environment():
#     """Setup Python virtual environment"""
#     print("🐍 Setting up virtual environment...")
    
#     if not Path("venv").exists():
#         if run_command("python -m venv venv", "Creating virtual environment"):
#             print("✅ Virtual environment created successfully!")
#             print("   Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
#         else:
#             print("⚠️  Could not create virtual environment automatically")
#             print("   Please create manually: python -m venv venv")
#     else:
#         print("✅ Virtual environment already exists!")

def verify_python_version():
    """Verify Python version compatibility"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   Please use Python 3.8 or higher")
        return False

def main():
    """Main setup function"""
    print("🚀 Synthetic Language Data Generation Pipeline Setup")
    print("=" * 60)
    
    # Check Python version
    if not verify_python_version():
        return
    
    # Create project structure
    setup_project_structure()
    
    # Create configuration files
    create_requirements_txt()
    # create_env_file() 
    # create_dockerignore()
    # create_readme()
    
    # # Setup virtual environment
    # setup_virtual_environment()
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("1. Activate virtual environment:")
    print("   source venv/bin/activate  # Linux/Mac")
    print("   venv\\Scripts\\activate     # Windows")
    print("\n2. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n3. Copy your script files to appropriate locations:")
    print("   cp your_scripts/data_ingestion.py src/ingestion.py")
    print("   cp your_scripts/text_generator.py src/generator.py")
    print("   cp your_scripts/validation_module.py src/validate.py")
    print("   cp your_scripts/serve.py src/serve.py")
    print("   cp your_scripts/prefect_pipeline.py pipelines/prefect/flows.py")
    print("\n4. Initialize the pipeline:")
    print("   python -c \"from src.pipeline_setup import initialize_project; initialize_project()\"")
    print("\n5. Run the demo:")
    print("   python demo_script.py --quick")
    
    print("\n🔧 Optional Docker Setup:")
    print("   docker-compose up --build -d")
    
    print(f"\n📁 Project created in: {Path.cwd()}")

if __name__ == "__main__":
    main()
