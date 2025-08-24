#!/usr/bin/env python3
"""
Create Expanded Sample Dataset for Testing Enhanced Features

This script creates a larger sample dataset to test the expanded feature engineering
without requiring Kaggle API setup. Simulates multiple data sources and larger scale.

Expected Impact: Demonstrates 25+ features vs current 20 features
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import random

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def create_expanded_sample_data():
    """Create expanded sample review dataset for testing."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 Creating expanded sample dataset for enhanced features...")
    
    # Expanded business types and locations
    business_types = [
        'Restaurant', 'Coffee Shop', 'Bakery', 'Fast Food', 'Fine Dining',
        'Retail Store', 'Auto Repair', 'Beauty Salon', 'Gym', 'Medical Clinic',
        'Law Firm', 'Consulting', 'Construction', 'Real Estate', 'Technology'
    ]
    
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
              'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
              'Austin', 'Jacksonville', 'San Francisco', 'Columbus', 'Charlotte']
    
    states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'FL', 'OH', 'NC']
    
    # Review templates for different sentiment levels
    review_templates = {
        5: [
            "Outstanding service and excellent quality! Highly recommend to everyone.",
            "Amazing experience! Staff was friendly and professional. Will definitely return.",
            "Perfect in every way. Great food, atmosphere, and service. Five stars!",
            "Exceptional quality and attention to detail. Best in the area!",
            "Wonderful experience from start to finish. Couldn't be happier!"
        ],
        4: [
            "Very good experience overall. Minor issues but would come back.",
            "Great service and good quality. Slightly pricey but worth it.",
            "Really enjoyed our visit. Good atmosphere and friendly staff.",
            "Solid choice with good value. Recommend for families.",
            "Nice place with consistent quality. Above average experience."
        ],
        3: [
            "Average experience, nothing special but decent quality.",
            "Okay service and food. Met expectations but nothing extraordinary.",
            "Acceptable quality for the price. Standard experience.",
            "Mixed experience - some good points, some areas for improvement.",
            "Fair quality and service. Would consider returning."
        ],
        2: [
            "Below average experience. Service was slow and quality lacking.",
            "Disappointed with the quality. Not worth the money.",
            "Poor service and mediocre results. Won't be returning.",
            "Subpar experience. Many better options available.",
            "Needs improvement in multiple areas. Not recommended."
        ],
        1: [
            "Terrible experience! Poor service and awful quality. Avoid at all costs!",
            "Worst service ever! Rude staff and terrible results. Never again!",
            "Complete waste of time and money. Absolutely horrible experience.",
            "Unacceptable quality and service. Management needs to address issues.",
            "Extremely disappointing. Would give zero stars if possible."
        ]
    }
    
    # Create expanded dataset
    businesses = []
    reviews = []
    
    # Generate businesses (500 instead of 10)
    num_businesses = 500
    
    for biz_id in range(num_businesses):
        business_type = random.choice(business_types)
        city = random.choice(cities)
        state = random.choice(states)
        
        # Create business name
        if business_type == 'Restaurant':
            names = ['Golden Dragon', 'Mama Mia', 'Spice Garden', 'Ocean View', 'Mountain Peak']
        elif business_type == 'Coffee Shop':
            names = ['Brew Masters', 'Coffee Corner', 'Bean There', 'Daily Grind', 'Espresso Bar']
        else:
            names = ['Elite', 'Premier', 'Quality', 'Professional', 'Expert']
        
        business_name = f"{random.choice(names)} {business_type}"
        
        # Business characteristics influence review patterns
        base_rating = random.uniform(2.0, 4.8)
        review_count = random.randint(5, 150)  # More realistic range
        
        businesses.append({
            'business_id': f'biz_{biz_id:03d}',
            'name': business_name,
            'city': city,
            'state': state,
            'categories': f"{business_type}, Service"
        })
        
        # Generate reviews for this business
        for review_idx in range(review_count):
            # Rating influenced by business base rating
            if base_rating >= 4.0:
                rating_weights = [0.05, 0.10, 0.15, 0.30, 0.40]  # Mostly 4-5 stars
            elif base_rating >= 3.0:
                rating_weights = [0.10, 0.20, 0.40, 0.20, 0.10]  # Mostly 2-4 stars
            else:
                rating_weights = [0.30, 0.30, 0.25, 0.10, 0.05]  # Mostly 1-3 stars
            
            rating = np.random.choice([1, 2, 3, 4, 5], p=rating_weights)
            
            # Generate review text
            base_text = random.choice(review_templates[rating])
            
            # Add variation to text length
            if random.random() < 0.3:  # 30% longer reviews
                additional_text = [
                    " The staff went above and beyond to ensure customer satisfaction.",
                    " The atmosphere was exactly what we were looking for.",
                    " Great attention to detail and professional service throughout.",
                    " Would definitely recommend to friends and family.",
                    " The quality exceeded our expectations in every way."
                ]
                base_text += random.choice(additional_text)
            
            # Generate review date (last 2 years)
            days_ago = random.randint(0, 730)
            review_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            reviews.append({
                'business_id': f'biz_{biz_id:03d}',
                'name': business_name,
                'city': city,
                'state': state,
                'stars': rating,
                'text': base_text,
                'review_id': f'rev_{biz_id:03d}_{review_idx:03d}',
                'date': review_date,
                'categories': f"{business_type}, Service",
                'data_source': 'expanded_sample',
                'source_priority': 1
            })
    
    # Convert to DataFrames
    businesses_df = pd.DataFrame(businesses)
    reviews_df = pd.DataFrame(reviews)
    
    # Save datasets
    data_dir = Path('data/external/free_reviews')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save expanded sample
    expanded_dir = data_dir / 'expanded_sample'
    expanded_dir.mkdir(exist_ok=True)
    
    reviews_df.to_csv(expanded_dir / 'expanded_reviews.csv', index=False)
    
    # Create unified format
    unified_file = data_dir / 'unified_business_reviews.csv'
    reviews_df.to_csv(unified_file, index=False)
    
    logger.info(f"✅ Created expanded sample dataset:")
    logger.info(f"   Businesses: {len(businesses):,}")
    logger.info(f"   Reviews: {len(reviews):,}")
    logger.info(f"   Average Reviews per Business: {len(reviews)/len(businesses):.1f}")
    logger.info(f"   Saved to: {unified_file}")
    
    # Show sample statistics
    rating_dist = reviews_df['stars'].value_counts().sort_index()
    logger.info(f"   Rating Distribution:")
    for rating, count in rating_dist.items():
        percentage = count / len(reviews) * 100
        logger.info(f"     {rating} stars: {count:,} ({percentage:.1f}%)")
    
    return len(businesses), len(reviews)

def main():
    """Main execution."""
    print("🚀 Creating Expanded Sample Dataset for Enhanced Features")
    print("=" * 60)
    
    num_businesses, num_reviews = create_expanded_sample_data()
    
    print(f"\n🎉 Sample dataset created successfully!")
    print(f"   Scale increase: 300 → {num_reviews:,} reviews")
    print(f"   Business count: 10 → {num_businesses:,} businesses")
    print(f"   Ready for enhanced preprocessing with 25+ features")
    print(f"\n💡 Next steps:")
    print(f"   1. Run: python scripts/external_data/run_expanded_preprocessing.py --run-full-pipeline")
    print(f"   2. Train models with expanded features")
    print(f"   3. Compare AUC improvement (target: 0.808 → 0.83+)")

if __name__ == "__main__":
    main()