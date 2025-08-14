#!/usr/bin/env python3
"""
Run data preprocessing pipeline for the AI-Powered Underwriting Engine.
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
import pickle
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from data_pipeline.preprocessing.data_preprocessor import LoanDataPreprocessor

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, 
                       preprocessor, output_dir="data/processed"):
    """Save processed datasets and preprocessor."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    datasets = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
    
    for name, data in datasets.items():
        filepath = output_path / f"{name}.parquet"
        if isinstance(data, pd.DataFrame):
            data.to_parquet(filepath, index=True)
        else:  # Series
            data.to_frame().to_parquet(filepath, index=True)
        print(f"Saved {name}: {filepath}")
    
    # Save preprocessor
    preprocessor_path = output_path / "preprocessor.pkl"
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Saved preprocessor: {preprocessor_path}")
    
    # Save feature summary
    summary = preprocessor.get_feature_summary()
    summary_path = output_path / "preprocessing_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("PREPROCESSING PIPELINE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary.items():
            if isinstance(value, list):
                f.write(f"{key}:\n")
                for item in value:
                    f.write(f"  - {item}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write(f"\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Saved summary: {summary_path}")

def main():
    """Main preprocessing function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting data preprocessing pipeline...")
    
    # Load raw data
    data_path = "scripts/data_acquisition/data/raw/sba/LoanData.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    logger.info(f"📥 Loading data from {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    logger.info(f"✅ Loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    # Initialize preprocessor
    logger.info("🔧 Initializing preprocessor...")
    preprocessor = LoanDataPreprocessor(
        missing_threshold=0.6,  # Drop features with >60% missing
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Run preprocessing
    logger.info("⚙️ Running preprocessing pipeline...")
    start_time = time.time()
    
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.fit_transform(df)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Preprocessing completed in {processing_time:.2f} seconds")
        
        # Print results summary
        print("\n📊 PREPROCESSING RESULTS")
        print("=" * 50)
        print(f"Original dataset: {len(df):,} rows, {len(df.columns)} columns")
        print(f"Processed features: {X_train.shape[1]} features")
        print(f"Target distribution:")
        print(f"  Train: {y_train.sum():,} defaults / {len(y_train):,} total ({y_train.mean():.2%})")
        print(f"  Val:   {y_val.sum():,} defaults / {len(y_val):,} total ({y_val.mean():.2%})")
        print(f"  Test:  {y_test.sum():,} defaults / {len(y_test):,} total ({y_test.mean():.2%})")
        
        print(f"\n🔧 PREPROCESSING SUMMARY")
        print("=" * 50)
        summary = preprocessor.get_feature_summary()
        for key, value in summary.items():
            if key != 'preprocessing_steps':
                print(f"{key}: {value}")
        
        # Save processed data
        logger.info("💾 Saving processed datasets...")
        save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, 
                           preprocessor, "data/processed")
        
        logger.info("🎉 Data preprocessing pipeline completed successfully!")
        
        # Quick data quality check
        print(f"\n✅ DATA QUALITY CHECK")
        print("=" * 50)
        print(f"Missing values in train set: {X_train.isnull().sum().sum()}")
        print(f"Infinite values in train set: {np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()}")
        print(f"Feature dtypes: {X_train.dtypes.value_counts().to_dict()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        raise e

if __name__ == "__main__":
    import numpy as np
    success = main()
    sys.exit(0 if success else 1)