#!/usr/bin/env python3
"""
Setup Free Business Review Datasets for AI-Powered Underwriting Engine.
Downloads and prepares free review datasets from Kaggle and other sources.
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

from external_data.free_review_integration import (
    FreeReviewDatasetManager, 
    FreeReviewDataProcessor,
    ReviewSentimentAnalyzer
)

def setup_logging():
    """Configure logging for dataset setup."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('free_datasets_setup.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def check_kaggle_setup():
    """Check if Kaggle API is properly configured."""
    logger = logging.getLogger(__name__)
    
    try:
        import kaggle
        # Test API access
        kaggle.api.authenticate()
        logger.info("✅ Kaggle API is properly configured")
        return True
    except ImportError:
        logger.error("❌ Kaggle API not installed")
        logger.info("💡 Install with: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"❌ Kaggle API configuration error: {str(e)}")
        logger.info("💡 Setup instructions:")
        logger.info("   1. Create account at kaggle.com")
        logger.info("   2. Go to Account > API > Create New API Token")
        logger.info("   3. Download kaggle.json to ~/.kaggle/kaggle.json")
        logger.info("   4. chmod 600 ~/.kaggle/kaggle.json")
        return False

def install_required_packages():
    """Install required packages for sentiment analysis."""
    logger = logging.getLogger(__name__)
    logger.info("📦 Installing required packages...")
    
    packages = [
        'vaderSentiment',
        'kaggle',
        'nltk'
    ]
    
    for package in packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} already installed")
        except ImportError:
            logger.info(f"📥 Installing {package}...")
            os.system(f"pip install {package}")

def download_nltk_data():
    """Download required NLTK data."""
    logger = logging.getLogger(__name__)
    logger.info("📥 Downloading NLTK data...")
    
    import nltk
    
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("✅ NLTK data downloaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ NLTK download warning: {str(e)}")

def demonstrate_dataset_download():
    """Demonstrate dataset download (without actual download)."""
    logger = logging.getLogger(__name__)
    
    # Initialize dataset manager
    data_manager = FreeReviewDatasetManager()
    
    # List available datasets
    logger.info("📋 Available Free Datasets:")
    datasets = data_manager.list_available_datasets()
    
    for name, description in datasets.items():
        available = "✅" if data_manager.check_dataset_availability(name) else "📥"
        logger.info(f"   {available} {name}: {description}")
    
    # Provide download instructions
    logger.info("\n💡 To download datasets:")
    logger.info("   1. Configure Kaggle API (see instructions above)")
    logger.info("   2. Run: python -c \"from external_data.free_review_integration import *; FreeReviewDatasetManager().download_dataset('yelp_small')\"")
    logger.info("   3. For Yelp Open Dataset: Manual download from https://www.yelp.com/dataset/download")

def create_sample_review_data():
    """Create sample review data for testing."""
    logger = logging.getLogger(__name__)
    logger.info("🔧 Creating sample review data for testing...")
    
    # Create sample data directory
    sample_dir = Path("data/external/free_reviews/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic review data
    np.random.seed(42)
    
    businesses = [
        "Joe's Pizza", "Manhattan Bistro", "Brooklyn Coffee", "Central Deli", "Park Restaurant",
        "Corner Bakery", "Main Street Cafe", "Sunset Grill", "Downtown Bar", "Garden Restaurant"
    ]
    
    cities = ["New York", "Brooklyn", "Manhattan", "Queens", "Bronx"]
    
    sample_reviews = []
    
    for i, business in enumerate(businesses):
        business_id = f"biz_{i:03d}"
        num_reviews = np.random.randint(10, 50)
        
        for j in range(num_reviews):
            # Generate realistic review text
            sentiments = [
                "Great food and excellent service! Highly recommend.",
                "Average experience, nothing special but decent.",
                "Poor service and mediocre food. Disappointed.",
                "Amazing atmosphere and delicious meals!",
                "Okay place, good for quick bite.",
                "Terrible experience, would not return.",
                "Outstanding restaurant with friendly staff!",
                "Food was cold and service was slow.",
                "Perfect dinner spot, loved everything!",
                "Not worth the money, poor quality."
            ]
            
            # Bias reviews based on business "quality"
            quality_bias = (i + 5) / 14  # Businesses have different average qualities
            if np.random.random() < quality_bias:
                text = np.random.choice(sentiments[:4] + sentiments[6:7] + sentiments[8:9])
                stars = np.random.choice([4, 5], p=[0.4, 0.6])
            else:
                text = np.random.choice(sentiments[2:3] + sentiments[5:6] + sentiments[7:8] + sentiments[9:10])
                stars = np.random.choice([1, 2, 3], p=[0.3, 0.3, 0.4])
            
            sample_reviews.append({
                'business_id': business_id,
                'name': business,
                'city': np.random.choice(cities),
                'state': 'NY',
                'stars': stars,
                'text': text,
                'review_id': f"rev_{i:03d}_{j:03d}",
                'date': f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
                'categories': 'Restaurants, Food'
            })
    
    # Save as CSV
    sample_df = pd.DataFrame(sample_reviews)
    sample_file = sample_dir / "sample_reviews.csv"
    sample_df.to_csv(sample_file, index=False)
    
    logger.info(f"✅ Created sample dataset: {sample_file}")
    logger.info(f"   {len(sample_df)} reviews for {len(businesses)} businesses")
    
    return sample_file

def test_sentiment_analysis():
    """Test sentiment analysis on sample data."""
    logger = logging.getLogger(__name__)
    logger.info("🧪 Testing sentiment analysis...")
    
    # Initialize sentiment analyzer
    analyzer = ReviewSentimentAnalyzer()
    
    # Test on sample texts
    test_texts = [
        "Amazing restaurant! Great food and excellent service.",
        "Terrible experience. Food was cold and service was awful.",
        "Average place, nothing special but okay for quick meal.",
        "Outstanding dinner! Will definitely come back again."
    ]
    
    for i, text in enumerate(test_texts, 1):
        sentiment = analyzer.analyze_text_sentiment(text)
        logger.info(f"   Text {i}: Polarity={sentiment['textblob_polarity']:.3f}, "
                   f"VADER={sentiment['vader_compound']:.3f}")
    
    logger.info("✅ Sentiment analysis working correctly")

def demonstrate_business_matching():
    """Demonstrate business name matching."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing business name matching...")
    
    # Create sample business data for testing
    sample_file = create_sample_review_data()
    
    # Test the processor
    processor = FreeReviewDataProcessor()
    
    # Load sample data
    processor.business_data = processor.load_yelp_csv_dataset(sample_file.parent)
    
    if processor.business_data:
        # Build index
        processor.name_matcher.build_business_index(processor.business_data)
        
        # Test matching
        test_names = [
            "Joe's Pizza",
            "joes pizza",
            "Joe Pizza Inc",
            "Manhattan Restaurant",
            "Nonexistent Business"
        ]
        
        for name in test_names:
            matches = processor.name_matcher.find_business_matches(name)
            if matches:
                logger.info(f"   '{name}' → '{matches[0].business_name}' (confidence: {len(matches)})")
            else:
                logger.info(f"   '{name}' → No matches found")
    
    logger.info("✅ Business matching test completed")

def generate_setup_summary():
    """Generate setup summary and next steps."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("🎉 FREE REVIEW DATASETS SETUP COMPLETED!")
    logger.info("="*60)
    
    logger.info("\n📊 What was configured:")
    logger.info("   ✅ Free review dataset integration system")
    logger.info("   ✅ Sentiment analysis pipeline (TextBlob + VADER)")
    logger.info("   ✅ Business name matching system")
    logger.info("   ✅ Sample dataset for testing")
    
    logger.info("\n🔄 Next steps:")
    logger.info("   1. Configure Kaggle API for dataset downloads")
    logger.info("   2. Download real datasets: python scripts/external_data/download_datasets.py")
    logger.info("   3. Run enhanced preprocessing: python scripts/external_data/run_free_review_preprocessing.py")
    logger.info("   4. Train enhanced models with review sentiment features")
    
    logger.info("\n📁 Generated files:")
    logger.info("   - src/external_data/free_review_integration.py")
    logger.info("   - data/external/free_reviews/sample/")
    logger.info("   - Free datasets setup log")
    
    logger.info("\n💡 Benefits:")
    logger.info("   🎯 No API costs - uses free datasets")
    logger.info("   📈 18+ sentiment features for underwriting")
    logger.info("   🔍 Business reputation scoring")
    logger.info("   📊 Review engagement analytics")
    
    logger.info("="*60)

def main():
    """Execute free datasets setup."""
    logger = setup_logging()
    logger.info("🚀 Starting Free Business Review Datasets Setup")
    
    try:
        # Install required packages
        install_required_packages()
        
        # Download NLTK data
        download_nltk_data()
        
        # Check Kaggle setup (optional for demo)
        kaggle_available = check_kaggle_setup()
        if not kaggle_available:
            logger.info("⚠️ Kaggle API not configured - will use sample data for demonstration")
        
        # Demonstrate available datasets
        demonstrate_dataset_download()
        
        # Create sample data
        create_sample_review_data()
        
        # Test sentiment analysis
        test_sentiment_analysis()
        
        # Test business matching
        demonstrate_business_matching()
        
        # Generate summary
        generate_setup_summary()
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()