#!/usr/bin/env python3
"""
Main data acquisition script that orchestrates the complete data download and validation process.
"""

import sys
import logging
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from download_sba_data import SBADataDownloader
from src.data_pipeline.validation.data_validator import DataValidator
import pandas as pd

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main data acquisition pipeline."""
    parser = argparse.ArgumentParser(description='Complete data acquisition pipeline for SBA dataset')
    parser.add_argument('--skip-download', action='store_true', help='Skip download and only validate existing data')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting complete data acquisition pipeline...")
    
    # Initialize components
    downloader = SBADataDownloader(data_dir=args.data_dir)
    validator = DataValidator()
    
    # Step 1: Download data (unless skipped)
    if not args.skip_download:
        logger.info("📥 Step 1: Downloading SBA dataset...")
        if not downloader.download_sba_dataset():
            logger.error("❌ Data download failed")
            return False
        
        # List and summarize downloaded files
        downloader.list_downloaded_files()
        downloader.create_data_summary()
    else:
        logger.info("⏭️  Skipping download step...")
    
    # Step 2: Validate data
    logger.info("🔍 Step 2: Validating downloaded data...")
    
    raw_dir = Path(args.data_dir) / "raw" / "sba"
    csv_files = list(raw_dir.glob("*.csv"))
    
    if not csv_files:
        logger.error("❌ No CSV files found for validation")
        return False
    
    validation_success = True
    
    for csv_file in csv_files:
        try:
            logger.info(f"Validating {csv_file.name}...")
            
            # Read the dataset
            df = pd.read_csv(csv_file)
            
            # Validate the dataset
            results = validator.validate_dataframe(df, csv_file.stem)
            
            # Print summary
            validator.print_summary(csv_file.stem)
            
        except Exception as e:
            logger.error(f"❌ Failed to validate {csv_file.name}: {str(e)}")
            validation_success = False
    
    # Step 3: Save validation report
    if validation_success:
        logger.info("📄 Step 3: Saving validation report...")
        report_path = raw_dir / "validation_report.json"
        validator.save_validation_report(report_path)
        
        logger.info("🎉 Data acquisition pipeline completed successfully!")
        logger.info(f"📁 Raw data: {raw_dir}")
        logger.info(f"📋 Validation report: {report_path}")
        
        return True
    else:
        logger.error("❌ Data acquisition pipeline failed during validation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)