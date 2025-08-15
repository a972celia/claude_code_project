#!/usr/bin/env python3
"""
Run Enhanced Preprocessing Pipeline with External Data Integration.
Tests the Yelp-enhanced preprocessing pipeline on existing loan data.
"""

import sys
import os
import logging
import pandas as pd
from pathlib import Path
import pickle

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from data_pipeline.preprocessing.enhanced_preprocessor import EnhancedLoanDataPreprocessor

def setup_logging():
    """Configure logging for enhanced preprocessing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('enhanced_preprocessing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_raw_data() -> pd.DataFrame:
    """Load the original raw loan dataset."""
    logger = logging.getLogger(__name__)
    
    # Try to load from multiple possible locations
    data_paths = [
        "data/raw/LoanData.csv",
        "scripts/data_acquisition/data/raw/sba/LoanData.csv", 
        "data/processed/raw_data_backup.csv"
    ]
    
    for path in data_paths:
        if Path(path).exists():
            logger.info(f"Loading raw data from: {path}")
            try:
                df = pd.read_csv(path)
                logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
                return df
            except Exception as e:
                logger.warning(f"Failed to load from {path}: {str(e)}")
                continue
    
    raise FileNotFoundError("Could not find raw loan dataset")

def run_enhanced_preprocessing(df: pd.DataFrame) -> tuple:
    """
    Run the enhanced preprocessing pipeline.
    
    Args:
        df: Raw loan dataset
        
    Returns:
        Tuple of processed datasets and preprocessor
    """
    logger = logging.getLogger(__name__)
    
    # Initialize enhanced preprocessor with Yelp integration
    logger.info("🚀 Initializing Enhanced Preprocessor with External Data...")
    
    preprocessor = EnhancedLoanDataPreprocessor(
        missing_threshold=0.6,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        enable_yelp=True  # Enable Yelp integration
    )
    
    # Run enhanced preprocessing
    logger.info("🔄 Running enhanced preprocessing pipeline...")
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_with_external_data(df)
    
    # Display results
    logger.info("✅ Enhanced preprocessing completed!")
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Validation set: {X_val.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Total features: {X_train.shape[1]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor

def analyze_external_features(X_train: pd.DataFrame):
    """Analyze the added external features."""
    logger = logging.getLogger(__name__)
    
    # Identify Yelp features
    yelp_features = [col for col in X_train.columns if col.startswith('yelp_')]
    
    if yelp_features:
        logger.info(f"📊 Analysis of {len(yelp_features)} Yelp features:")
        
        for feature in yelp_features[:10]:  # Show first 10 features
            values = X_train[feature]
            logger.info(f"  {feature}:")
            logger.info(f"    Mean: {values.mean():.4f}")
            logger.info(f"    Std:  {values.std():.4f}")
            logger.info(f"    Range: [{values.min():.3f}, {values.max():.3f}]")
        
        # Check correlations with key loan features
        if 'Interest' in X_train.columns:
            correlations = X_train[yelp_features + ['Interest']].corr()['Interest'].sort_values(key=abs, ascending=False)[1:]
            
            logger.info("\n🔗 Top Yelp features correlated with Interest rate:")
            for feature, corr in correlations.head(5).items():
                logger.info(f"  {feature}: {corr:.4f}")
    else:
        logger.warning("⚠️ No Yelp features found in processed dataset")

def save_enhanced_data(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor):
    """Save enhanced processed data."""
    logger = logging.getLogger(__name__)
    
    # Create enhanced data directory
    enhanced_dir = Path("data/processed_enhanced")
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("💾 Saving enhanced processed datasets...")
    
    # Save datasets
    X_train.to_parquet(enhanced_dir / "X_train_enhanced.parquet")
    X_val.to_parquet(enhanced_dir / "X_val_enhanced.parquet")
    X_test.to_parquet(enhanced_dir / "X_test_enhanced.parquet")
    y_train.to_frame().to_parquet(enhanced_dir / "y_train_enhanced.parquet")
    y_val.to_frame().to_parquet(enhanced_dir / "y_val_enhanced.parquet")
    y_test.to_frame().to_parquet(enhanced_dir / "y_test_enhanced.parquet")
    
    # Save enhanced preprocessor
    with open(enhanced_dir / "enhanced_preprocessor.pkl", 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Save feature list
    feature_info = {
        'total_features': X_train.shape[1],
        'numeric_features': len(preprocessor.numeric_features),
        'categorical_features': len(preprocessor.categorical_features),
        'yelp_features': len([f for f in X_train.columns if f.startswith('yelp_')]),
        'feature_names': X_train.columns.tolist()
    }
    
    with open(enhanced_dir / "enhanced_features.json", 'w') as f:
        import json
        json.dump(feature_info, f, indent=2)
    
    # Save enhanced summary
    with open(enhanced_dir / "enhanced_preprocessing_summary.txt", 'w') as f:
        f.write(preprocessor.generate_enhanced_summary())
    
    logger.info(f"✅ Enhanced datasets saved to: {enhanced_dir}")
    return enhanced_dir

def main():
    """Execute enhanced preprocessing pipeline."""
    logger = setup_logging()
    logger.info("🌟 Starting Enhanced Preprocessing with External Data Integration")
    
    try:
        # Load raw data
        df = load_raw_data()
        
        # Run enhanced preprocessing
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = run_enhanced_preprocessing(df)
        
        # Analyze external features
        analyze_external_features(X_train)
        
        # Save enhanced data
        enhanced_dir = save_enhanced_data(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor)
        
        # Display summary
        print("\n" + "="*70)
        print("🎉 ENHANCED PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📂 Enhanced datasets saved to: {enhanced_dir}")
        print(f"📊 Total features: {X_train.shape[1]} (including external data)")
        print(f"🎯 Ready for enhanced model training with external data!")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Enhanced preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()