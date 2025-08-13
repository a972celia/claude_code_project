#!/usr/bin/env python3
"""
SBA loan dataset download script for the AI-Powered Underwriting Engine.
"""

import os
import sys
import logging
import zipfile
import pandas as pd
from pathlib import Path
from typing import Optional
import kaggle
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class SBADataDownloader:
    """Class to handle SBA dataset download and processing."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "sba"
        self.processed_dir = self.data_dir / "processed" / "sba"
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def authenticate_kaggle(self):
        """Authenticate with Kaggle API."""
        try:
            kaggle.api.authenticate()
            self.logger.info("✅ Kaggle authentication successful")
            return True
        except Exception as e:
            self.logger.error(f"❌ Kaggle authentication failed: {str(e)}")
            self.logger.error("Please run 'python scripts/data_acquisition/setup_kaggle.py' first")
            return False
    
    def download_sba_dataset(self, dataset_name: str = "mirbektoktogaraev/small-business-administration-sba-loans"):
        """
        Download SBA loan dataset from Kaggle.
        
        Args:
            dataset_name: Kaggle dataset identifier
        """
        if not self.authenticate_kaggle():
            return False
        
        try:
            self.logger.info(f"Downloading SBA dataset: {dataset_name}")
            
            # Download dataset
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(self.raw_dir),
                unzip=True
            )
            
            self.logger.info(f"✅ Dataset downloaded to {self.raw_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to download dataset: {str(e)}")
            return False
    
    def list_downloaded_files(self):
        """List all downloaded files."""
        files = list(self.raw_dir.glob("*"))
        self.logger.info(f"Downloaded files in {self.raw_dir}:")
        for file in files:
            self.logger.info(f"  - {file.name} ({file.stat().st_size / 1024 / 1024:.2f} MB)")
        return files
    
    def validate_dataset(self):
        """Validate the downloaded dataset."""
        self.logger.info("Validating SBA dataset...")
        
        # Look for CSV files
        csv_files = list(self.raw_dir.glob("*.csv"))
        
        if not csv_files:
            self.logger.error("❌ No CSV files found in downloaded data")
            return False
        
        validation_results = {}
        
        for csv_file in csv_files:
            try:
                # Read a sample of the data
                df = pd.read_csv(csv_file, nrows=1000)
                
                validation_results[csv_file.name] = {
                    "rows_sample": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns),
                    "file_size_mb": csv_file.stat().st_size / 1024 / 1024
                }
                
                self.logger.info(f"✅ {csv_file.name}: {len(df.columns)} columns, sample size: {len(df)} rows")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to validate {csv_file.name}: {str(e)}")
                validation_results[csv_file.name] = {"error": str(e)}
        
        return validation_results
    
    def create_data_summary(self):
        """Create a summary of the downloaded data."""
        csv_files = list(self.raw_dir.glob("*.csv"))
        
        summary = {
            "download_date": pd.Timestamp.now().isoformat(),
            "total_files": len(csv_files),
            "files": {}
        }
        
        for csv_file in csv_files:
            try:
                # Get basic info without loading full dataset
                df_sample = pd.read_csv(csv_file, nrows=1000)
                
                # Try to get full row count efficiently
                with open(csv_file, 'r') as f:
                    total_rows = sum(1 for line in f) - 1  # Subtract header
                
                summary["files"][csv_file.name] = {
                    "total_rows": total_rows,
                    "columns": len(df_sample.columns),
                    "file_size_mb": round(csv_file.stat().st_size / 1024 / 1024, 2),
                    "column_names": list(df_sample.columns),
                    "sample_data_types": df_sample.dtypes.astype(str).to_dict()
                }
                
            except Exception as e:
                summary["files"][csv_file.name] = {"error": str(e)}
        
        # Save summary
        summary_path = self.raw_dir / "dataset_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📊 Dataset summary saved to {summary_path}")
        return summary

def main():
    """Main download function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting SBA dataset download...")
    
    # Initialize downloader
    downloader = SBADataDownloader()
    
    # Download dataset
    if downloader.download_sba_dataset():
        # List downloaded files
        downloader.list_downloaded_files()
        
        # Validate dataset
        validation_results = downloader.validate_dataset()
        
        # Create summary
        summary = downloader.create_data_summary()
        
        logger.info("🎉 SBA dataset download and validation completed successfully!")
        logger.info("📁 Data location: data/raw/sba/")
        logger.info("📋 Summary: data/raw/sba/dataset_summary.json")
        
        return True
    else:
        logger.error("❌ SBA dataset download failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)