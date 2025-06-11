#!/usr/bin/env python3
"""
Utility script for MediBot project management and validation.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

from config import *

logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def check_environment() -> Dict[str, Any]:
    """
    Check if the environment is properly configured.
    
    Returns:
        Dictionary with environment status
    """
    status = {
        "valid": True,
        "issues": [],
        "warnings": []
    }
    
    # Check Python version
    if sys.version_info < (3, 11):
        status["valid"] = False
        status["issues"].append(f"Python 3.11+ required, found {sys.version}")
    
    # Check HF_TOKEN
    if not HF_TOKEN or HF_TOKEN == "your_hugging_face_token_here":
        status["valid"] = False
        status["issues"].append("HF_TOKEN not properly set in environment variables")
    
    # Check data directory
    if not DATA_PATH.exists():
        status["warnings"].append(f"Data directory not found: {DATA_PATH}")
    elif not list(DATA_PATH.glob("*.pdf")):
        status["warnings"].append("No PDF files found in data directory")
    
    # Check vectorstore
    if not VECTORSTORE_PATH.exists():
        status["warnings"].append(f"Vector store not found: {VECTORSTORE_PATH}")
    
    return status


def validate_dependencies() -> bool:
    """
    Validate that all required dependencies are installed.
    
    Returns:
        True if all dependencies are available
    """
    required_modules = [
        "streamlit",
        "langchain",
        "langchain_community",
        "langchain_huggingface",
        "faiss",
        "sentence_transformers",
        "pypdf"
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module.replace("-", "_"))
        except ImportError:
            missing.append(module)
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.info("Install missing dependencies with: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are available")
    return True


def setup_project() -> bool:
    """
    Set up the project directories and files.
    
    Returns:
        True if setup is successful
    """
    try:
        # Create directories
        DATA_PATH.mkdir(exist_ok=True)
        VECTORSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Project directories created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to set up project: {str(e)}")
        return False


def get_project_stats() -> Dict[str, Any]:
    """
    Get statistics about the project.
    
    Returns:
        Dictionary with project statistics
    """
    stats = {
        "pdf_files": 0,
        "vectorstore_exists": False,
        "vectorstore_size": 0
    }
    
    # Count PDF files
    if DATA_PATH.exists():
        stats["pdf_files"] = len(list(DATA_PATH.glob("*.pdf")))
    
    # Check vectorstore
    if VECTORSTORE_PATH.exists():
        stats["vectorstore_exists"] = True
        # Calculate vectorstore size
        total_size = 0
        for file_path in VECTORSTORE_PATH.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        stats["vectorstore_size"] = total_size
    
    return stats


def main():
    """Main function for project utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MediBot Project Utilities")
    parser.add_argument("--check", action="store_true", help="Check environment")
    parser.add_argument("--setup", action="store_true", help="Set up project")
    parser.add_argument("--stats", action="store_true", help="Show project statistics")
    parser.add_argument("--validate", action="store_true", help="Validate dependencies")
    
    args = parser.parse_args()
    
    if args.check:
        print("\n🔍 Checking Environment...")
        print("=" * 40)
        
        status = check_environment()
        
        if status["valid"]:
            print("✅ Environment is properly configured")
        else:
            print("❌ Environment issues found:")
            for issue in status["issues"]:
                print(f"  - {issue}")
        
        if status["warnings"]:
            print("\n⚠️  Warnings:")
            for warning in status["warnings"]:
                print(f"  - {warning}")
    
    elif args.setup:
        print("\n🛠️  Setting up project...")
        if setup_project():
            print("✅ Project setup completed")
        else:
            print("❌ Project setup failed")
    
    elif args.stats:
        print("\n📊 Project Statistics")
        print("=" * 40)
        
        stats = get_project_stats()
        print(f"PDF files: {stats['pdf_files']}")
        print(f"Vector store exists: {'Yes' if stats['vectorstore_exists'] else 'No'}")
        
        if stats['vectorstore_exists']:
            size_mb = stats['vectorstore_size'] / (1024 * 1024)
            print(f"Vector store size: {size_mb:.2f} MB")
    
    elif args.validate:
        print("\n🔧 Validating Dependencies...")
        print("=" * 40)
        
        if validate_dependencies():
            print("✅ All dependencies are installed")
        else:
            print("❌ Some dependencies are missing")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
