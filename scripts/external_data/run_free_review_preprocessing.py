#!/usr/bin/env python3
"""
Enhanced Preprocessing with Free Review Dataset Integration.
Integrates free business review sentiment data with loan preprocessing pipeline.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from external_data.free_review_integration import FreeReviewDataProcessor
from data_pipeline.preprocessing.data_preprocessor import LoanDataPreprocessor

def setup_logging():
    """Configure logging for enhanced preprocessing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('free_review_preprocessing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_raw_loan_data() -> pd.DataFrame:
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

def create_synthetic_business_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic business names and locations to loan data for demonstration.
    In production, this would come from loan application forms.
    """
    logger = logging.getLogger(__name__)
    logger.info("🔧 Creating synthetic business information for demonstration...")
    
    # Business name templates
    business_types = [
        "Restaurant", "Cafe", "Bakery", "Deli", "Bistro", "Grill", "Bar", "Pizza",
        "Coffee Shop", "Steakhouse", "Sushi", "Taco", "Burger", "BBQ", "Seafood"
    ]
    
    locations = ["Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"]
    
    np.random.seed(42)
    n_samples = len(df)
    
    # Create realistic business names
    business_names = []
    business_locations = []
    
    for i in range(n_samples):
        # Generate business name
        if np.random.random() < 0.3:
            # Owner's name style
            owner_names = ["Joe's", "Mike's", "Anna's", "Tony's", "Maria's", "Sam's"]
            name = f"{np.random.choice(owner_names)} {np.random.choice(business_types)}"
        else:
            # Descriptive name style
            adjectives = ["Central", "Corner", "Main Street", "Downtown", "Park", "Golden", "Royal"]
            name = f"{np.random.choice(adjectives)} {np.random.choice(business_types)}"
        
        business_names.append(name)
        
        # Add location
        location = f"{np.random.choice(locations)}, NY"
        business_locations.append(location)
    
    # Add to dataframe
    df_enhanced = df.copy()
    df_enhanced['business_name'] = business_names
    df_enhanced['business_location'] = business_locations
    
    logger.info(f"✅ Added synthetic business information to {len(df_enhanced)} records")
    return df_enhanced

def load_and_process_review_data() -> FreeReviewDataProcessor:
    """Load and process review data."""
    logger = logging.getLogger(__name__)
    
    # Initialize processor
    processor = FreeReviewDataProcessor()
    
    # Try to load a real dataset first
    datasets_to_try = ['yelp_small', 'yelp_polarity']
    
    for dataset_name in datasets_to_try:
        logger.info(f"Attempting to load dataset: {dataset_name}")
        if processor.load_dataset(dataset_name):
            logger.info(f"✅ Successfully loaded {dataset_name}")
            return processor
    
    # Fall back to sample data
    logger.info("Real datasets not available, using sample data...")
    sample_path = Path("data/external/free_reviews/sample")
    
    if sample_path.exists():
        processor.business_data = processor.load_yelp_csv_dataset(sample_path)
        if processor.business_data:
            processor.name_matcher.build_business_index(processor.business_data)
            logger.info(f"✅ Loaded sample dataset with {len(processor.business_data)} businesses")
            return processor
    
    logger.error("❌ No review data available")
    return None

def run_enhanced_preprocessing():
    """Run preprocessing with free review integration."""
    logger = logging.getLogger(__name__)
    
    # Load raw loan data
    logger.info("📂 Loading raw loan data...")
    raw_df = load_raw_loan_data()
    
    # Add synthetic business information for demonstration
    df_with_business = create_synthetic_business_info(raw_df)
    
    # Load review data
    logger.info("📊 Loading review dataset...")
    review_processor = load_and_process_review_data()
    
    if not review_processor:
        logger.error("❌ Could not load review data. Run setup_free_datasets.py first.")
        return None
    
    # Enhance loan data with review sentiment
    logger.info("🔗 Integrating review sentiment data with loan applications...")
    
    # Take a sample for faster processing
    sample_size = min(10000, len(df_with_business))
    df_sample = df_with_business.sample(n=sample_size, random_state=42)
    
    enhanced_df = review_processor.enhance_loan_data(
        df_sample,
        business_name_col='business_name',
        location_col='business_location'
    )
    
    # Run standard preprocessing
    logger.info("🔄 Running standard loan preprocessing...")
    preprocessor = LoanDataPreprocessor(
        missing_threshold=0.6,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.fit_transform(enhanced_df)
    
    # Analyze review features
    review_features = [col for col in X_train.columns if col.startswith('review_')]
    
    logger.info(f"✅ Enhanced preprocessing completed!")
    logger.info(f"   Original features: {len(X_train.columns) - len(review_features)}")
    logger.info(f"   Review features: {len(review_features)}")
    logger.info(f"   Total features: {len(X_train.columns)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, review_features

def analyze_review_features(X_train: pd.DataFrame, review_features: list):
    """Analyze the integrated review features."""
    logger = logging.getLogger(__name__)
    
    if not review_features:
        logger.warning("⚠️ No review features found")
        return
    
    logger.info("📈 Review Feature Analysis:")
    
    # Feature statistics
    for feature in review_features[:10]:  # Show first 10
        values = X_train[feature]
        non_zero_pct = (values != 0).mean() * 100
        
        logger.info(f"   {feature}:")
        logger.info(f"     Mean: {values.mean():.4f}")
        logger.info(f"     Std:  {values.std():.4f}")
        logger.info(f"     Non-zero: {non_zero_pct:.1f}%")
    
    # Data availability analysis
    data_source_counts = X_train['review_data_source'].value_counts() if 'review_data_source' in X_train.columns else {}
    logger.info(f"\n📊 Review Data Coverage:")
    for source, count in data_source_counts.items():
        pct = count / len(X_train) * 100
        logger.info(f"   {source}: {count:,} ({pct:.1f}%)")
    
    # Correlation with loan features
    if 'Interest' in X_train.columns:
        correlations = X_train[review_features + ['Interest']].corr()['Interest'].sort_values(key=abs, ascending=False)
        
        logger.info(f"\n🔗 Top Review Features Correlated with Interest Rate:")
        for feature, corr in correlations[1:6].items():  # Top 5 excluding Interest itself
            logger.info(f"   {feature}: {corr:.4f}")

def save_enhanced_data(X_train, X_val, X_test, y_train, y_val, y_test, review_features):
    """Save enhanced data with review features."""
    logger = logging.getLogger(__name__)
    
    # Create enhanced data directory
    enhanced_dir = Path("data/processed_enhanced_free")
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("💾 Saving enhanced datasets with free review features...")
    
    # Save datasets
    X_train.to_parquet(enhanced_dir / "X_train_enhanced_free.parquet")
    X_val.to_parquet(enhanced_dir / "X_val_enhanced_free.parquet")
    X_test.to_parquet(enhanced_dir / "X_test_enhanced_free.parquet")
    y_train.to_frame().to_parquet(enhanced_dir / "y_train_enhanced_free.parquet")
    y_val.to_frame().to_parquet(enhanced_dir / "y_val_enhanced_free.parquet")
    y_test.to_frame().to_parquet(enhanced_dir / "y_test_enhanced_free.parquet")
    
    # Save feature information
    feature_info = {
        'total_features': len(X_train.columns),
        'review_features': len(review_features),
        'original_features': len(X_train.columns) - len(review_features),
        'review_feature_names': review_features,
        'all_feature_names': X_train.columns.tolist(),
        'data_source': 'free_review_datasets',
        'processing_date': pd.Timestamp.now().isoformat()
    }
    
    with open(enhanced_dir / "enhanced_features_free.json", 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # Save summary
    summary = f"""FREE REVIEW DATASET INTEGRATION SUMMARY
=====================================================

Dataset Information:
- Total samples: {len(X_train):,}
- Total features: {len(X_train.columns)}
- Original loan features: {len(X_train.columns) - len(review_features)}
- Review sentiment features: {len(review_features)}

Review Features Generated:
{chr(10).join(f"  - {feature}" for feature in review_features)}

Data Sources:
- Free business review datasets (Kaggle/sample)
- TextBlob and VADER sentiment analysis
- Business name matching and fuzzy search

Benefits:
- No API costs
- Offline processing
- Sentiment-driven risk assessment
- Business reputation scoring

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(enhanced_dir / "free_review_integration_summary.txt", 'w') as f:
        f.write(summary)
    
    logger.info(f"✅ Enhanced datasets saved to: {enhanced_dir}")
    return enhanced_dir

def main():
    """Execute enhanced preprocessing with free review data."""
    logger = setup_logging()
    logger.info("🚀 Starting Enhanced Preprocessing with Free Review Data")
    
    try:
        # Run enhanced preprocessing
        result = run_enhanced_preprocessing()
        
        if result is None:
            logger.error("❌ Enhanced preprocessing failed")
            return
        
        X_train, X_val, X_test, y_train, y_val, y_test, review_features = result
        
        # Analyze review features
        analyze_review_features(X_train, review_features)
        
        # Save enhanced data
        enhanced_dir = save_enhanced_data(X_train, X_val, X_test, y_train, y_val, y_test, review_features)
        
        # Display summary
        print("\n" + "="*70)
        print("🎉 FREE REVIEW DATA INTEGRATION COMPLETED!")
        print("="*70)
        print(f"📂 Enhanced datasets: {enhanced_dir}")
        print(f"📊 Total features: {len(X_train.columns)} (+{len(review_features)} review features)")
        print(f"🎯 Ready for enhanced model training!")
        print(f"💰 Cost: $0 (using free datasets)")
        print("="*70)
        
    except Exception as e:
        logger.error(f"❌ Enhanced preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()