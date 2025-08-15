#!/usr/bin/env python3
"""
Demonstration of Yelp Data Integration for AI-Powered Underwriting Engine.
Shows how to collect business data and enhance underwriting models.
"""

import sys
import os
import logging
import pandas as pd
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from external_data.yelp_integration import YelpDataPipeline, YelpDataCollector, YelpFeatureEngineer
from config.yelp_config import yelp_config

def setup_logging():
    """Configure logging for demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def demo_business_search(api_key: str):
    """Demonstrate business search functionality."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Demonstrating Yelp business search...")
    
    collector = YelpDataCollector(api_key)
    
    # Search for restaurants in NYC
    businesses = collector.search_businesses(
        location="New York, NY",
        categories="restaurants",
        limit=5
    )
    
    if businesses:
        logger.info(f"Found {len(businesses)} businesses:")
        for i, business in enumerate(businesses[:3], 1):
            logger.info(f"{i}. {business.get('name')} - Rating: {business.get('rating')}/5 "
                       f"({business.get('review_count')} reviews)")
    else:
        logger.warning("No businesses found")
    
    return businesses

def demo_business_analysis(api_key: str, business_id: str):
    """Demonstrate detailed business analysis."""
    logger = logging.getLogger(__name__)
    logger.info(f"🏢 Analyzing business: {business_id}")
    
    collector = YelpDataCollector(api_key)
    feature_engineer = YelpFeatureEngineer()
    
    # Collect comprehensive business data
    yelp_data = collector.collect_business_data(business_id)
    
    if yelp_data:
        logger.info(f"Business: {yelp_data.name}")
        logger.info(f"Rating: {yelp_data.rating}/5 ({yelp_data.review_count} reviews)")
        logger.info(f"Categories: {', '.join(yelp_data.categories)}")
        logger.info(f"Price Level: {yelp_data.price}")
        
        # Show sentiment analysis
        sentiment = yelp_data.sentiment_scores
        logger.info(f"Sentiment Analysis:")
        logger.info(f"  Average Polarity: {sentiment.get('avg_polarity', 0):.3f}")
        logger.info(f"  Average Subjectivity: {sentiment.get('avg_subjectivity', 0):.3f}")
        
        # Generate features
        features = feature_engineer.create_business_features(yelp_data)
        logger.info(f"Generated {len(features)} features for underwriting model")
        
        # Show key risk indicators
        logger.info("Key Risk Indicators:")
        logger.info(f"  Reputation Score: {features.get('yelp_reputation_score', 0):.3f}")
        logger.info(f"  Poor Rating: {'Yes' if features.get('yelp_poor_rating') else 'No'}")
        logger.info(f"  Negative Sentiment: {'Yes' if features.get('yelp_negative_sentiment') else 'No'}")
        logger.info(f"  High Risk Category: {'Yes' if features.get('yelp_high_risk_category') else 'No'}")
        
        return features
    else:
        logger.error("Failed to collect business data")
        return None

def demo_loan_enhancement():
    """Demonstrate loan dataset enhancement with Yelp data."""
    logger = logging.getLogger(__name__)
    logger.info("💼 Demonstrating loan dataset enhancement...")
    
    # Create sample loan data
    sample_loans = pd.DataFrame({
        'loan_id': ['L001', 'L002', 'L003'],
        'business_name': ['Joe\'s Pizza', 'Manhattan Bistro', 'Brooklyn Coffee'],
        'business_location': ['New York, NY', 'New York, NY', 'Brooklyn, NY'],
        'loan_amount': [50000, 100000, 25000],
        'interest_rate': [6.5, 5.8, 7.2]
    })
    
    logger.info(f"Sample dataset: {len(sample_loans)} loans")
    print(sample_loans)
    
    # Note: This would require actual API key to run
    logger.info("\n📋 To run actual enhancement, configure Yelp API key and run:")
    logger.info("pipeline = YelpDataPipeline(api_key)")
    logger.info("enhanced_df = pipeline.enhance_loan_data(sample_loans)")
    
    return sample_loans

def main():
    """Run Yelp integration demonstration."""
    logger = setup_logging()
    logger.info("🚀 Starting Yelp Integration Demonstration")
    
    # Check configuration
    if not yelp_config.is_configured():
        logger.warning("⚠️  Yelp API key not configured")
        print(yelp_config.setup_instructions())
        
        # Show what features would be created
        logger.info("\n📊 Example features that would be generated:")
        feature_engineer = YelpFeatureEngineer()
        
        # Create mock Yelp data for demonstration
        from external_data.yelp_integration import YelpBusinessData
        mock_data = YelpBusinessData(
            id="example-business",
            name="Example Restaurant",
            rating=4.2,
            review_count=156,
            categories=["Italian", "Restaurants"],
            price="$$",
            location={"city": "New York"},
            phone="+1-212-555-0123",
            display_phone="(212) 555-0123",
            url="https://yelp.com/biz/example",
            is_closed=False,
            coordinates={"latitude": 40.7128, "longitude": -74.0060},
            photos=["photo1.jpg", "photo2.jpg"],
            hours=None,
            reviews=[
                {"text": "Great food and service!", "rating": 5},
                {"text": "Good atmosphere", "rating": 4},
                {"text": "Average experience", "rating": 3}
            ],
            sentiment_scores={
                "avg_polarity": 0.25,
                "avg_subjectivity": 0.6,
                "sentiment_variance": 0.1,
                "review_count_with_text": 3
            }
        )
        
        features = feature_engineer.create_business_features(mock_data)
        logger.info("\nGenerated Yelp Features:")
        for feature, value in features.items():
            logger.info(f"  {feature}: {value}")
        
        return
    
    try:
        api_key = yelp_config.get_api_key()
        
        # Demo 1: Business search
        businesses = demo_business_search(api_key)
        
        if businesses:
            # Demo 2: Detailed analysis of first business
            first_business = businesses[0]
            demo_business_analysis(api_key, first_business['id'])
        
        # Demo 3: Loan dataset enhancement (mock)
        demo_loan_enhancement()
        
        logger.info("✅ Demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {str(e)}")

if __name__ == "__main__":
    main()