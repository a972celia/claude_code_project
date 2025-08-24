#!/usr/bin/env python3
"""
Expand Free Review Datasets for Enhanced AI-Powered Underwriting

This script downloads and integrates multiple large-scale free business review datasets
to significantly improve model performance beyond the current 0.808 AUC.

Target Datasets:
1. Kaggle Yelp Sentiment Dataset (598K reviews)
2. Kaggle Yelp Reviews Full (650K reviews) 
3. Google Business Reviews Dataset
4. Additional Kaggle Restaurant Review Collections

Expected Impact: 2-5% AUC improvement (0.808 → 0.83-0.85)
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import json
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Optional, Union

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.external_data.free_review_integration import (
    FreeReviewDatasetManager, 
    FreeReviewDataProcessor,
    ReviewSentimentAnalyzer
)

class ExpandedDatasetManager:
    """Manager for downloading and processing multiple large-scale free datasets."""
    
    def __init__(self, data_dir: str = "data/external/free_reviews"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Expanded dataset catalog
        self.dataset_catalog = {
            'yelp_review_sentiment': {
                'kaggle_ref': 'ilhamfp31/yelp-review-dataset',
                'description': 'Yelp Review Sentiment Dataset (598K reviews)',
                'size': '598K reviews',
                'business_coverage': 'High',
                'files': ['yelp_reviews.csv']
            },
            'yelp_reviews_full': {
                'kaggle_ref': 'omkarsabnis/yelp-reviews-dataset',
                'description': 'Yelp Reviews Dataset (650K reviews)',
                'size': '650K reviews', 
                'business_coverage': 'High',
                'files': ['yelp.csv']
            },
            'restaurant_reviews_huge': {
                'kaggle_ref': 'snap/amazon-fine-food-reviews',
                'description': 'Amazon Fine Food Reviews (568K reviews)',
                'size': '568K reviews',
                'business_coverage': 'Food/Restaurant',
                'files': ['Reviews.csv']
            },
            'google_business_reviews': {
                'kaggle_ref': 'bharatnatrayn/google-my-business-reviews',
                'description': 'Google My Business Reviews Dataset',
                'size': '100K+ reviews',
                'business_coverage': 'Mixed Business Types',
                'files': ['google_reviews.csv']
            },
            'tripadvisor_restaurant_reviews': {
                'kaggle_ref': 'damienbeneschi/krakow-ta-restaurans-data-raw',
                'description': 'TripAdvisor Restaurant Reviews',
                'size': '50K+ reviews',
                'business_coverage': 'Restaurants',
                'files': ['tripadvisor_reviews.csv']
            },
            'business_sentiment_large': {
                'kaggle_ref': 'kritanjalijain/amazon-reviews',
                'description': 'Large Business Review Collection',
                'size': '3M+ reviews',
                'business_coverage': 'Mixed Business',
                'files': ['train.csv']
            }
        }
        
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup enhanced logging for dataset expansion."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # File handler
            log_file = self.data_dir.parent.parent.parent / 'dataset_expansion.log'
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.DEBUG)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    def list_available_datasets(self):
        """Display all available datasets with details."""
        print("\n🗂️  Available Free Business Review Datasets")
        print("=" * 80)
        
        for name, info in self.dataset_catalog.items():
            print(f"\n📊 {name}")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   Business Coverage: {info['business_coverage']}")
            print(f"   Kaggle Reference: {info['kaggle_ref']}")
            
            # Check if already downloaded
            dataset_path = self.data_dir / name
            if dataset_path.exists():
                print(f"   Status: ✅ Downloaded")
            else:
                print(f"   Status: ⏳ Available for download")
    
    def download_large_dataset(self, dataset_name: str, force_redownload: bool = False) -> bool:
        """
        Download a large-scale dataset with progress tracking.
        
        Args:
            dataset_name: Name of dataset to download
            force_redownload: Force redownload even if exists
            
        Returns:
            Success status
        """
        if dataset_name not in self.dataset_catalog:
            self.logger.error(f"❌ Unknown dataset: {dataset_name}")
            return False
        
        dataset_info = self.dataset_catalog[dataset_name]
        dataset_path = self.data_dir / dataset_name
        
        # Check if already exists
        if dataset_path.exists() and not force_redownload:
            self.logger.info(f"✅ Dataset {dataset_name} already exists")
            return True
        
        # Create dataset directory
        dataset_path.mkdir(exist_ok=True)
        
        try:
            import kaggle
            
            self.logger.info(f"🚀 Downloading large dataset: {dataset_name}")
            self.logger.info(f"   Size: {dataset_info['size']}")
            self.logger.info(f"   This may take several minutes...")
            
            # Download with progress
            kaggle.api.dataset_download_files(
                dataset_info['kaggle_ref'],
                path=dataset_path,
                unzip=True,
                quiet=False
            )
            
            self.logger.info(f"✅ Successfully downloaded: {dataset_name}")
            
            # Verify download
            csv_files = list(dataset_path.glob("*.csv"))
            if csv_files:
                sample_df = pd.read_csv(csv_files[0], nrows=5)
                self.logger.info(f"   Sample data preview: {sample_df.shape}")
                self.logger.info(f"   Columns: {list(sample_df.columns)}")
            
            return True
            
        except ImportError:
            self.logger.error("❌ Kaggle API not installed. Run: pip install kaggle")
            self.logger.info("   Alternative: Manual download from kaggle.com")
            return False
        except Exception as e:
            self.logger.error(f"❌ Failed to download {dataset_name}: {str(e)}")
            return False
    
    def download_priority_datasets(self) -> Dict[str, bool]:
        """Download the highest-impact datasets for model improvement."""
        priority_datasets = [
            'yelp_review_sentiment',  # 598K reviews - highest quality
            'yelp_reviews_full',      # 650K reviews - broad coverage
            'google_business_reviews', # Google data - different source
            'restaurant_reviews_huge'  # 568K food reviews - sector coverage
        ]
        
        results = {}
        total_reviews = 0
        
        self.logger.info("🎯 Downloading priority datasets for maximum model impact...")
        
        for dataset_name in priority_datasets:
            self.logger.info(f"\n📥 Processing: {dataset_name}")
            success = self.download_large_dataset(dataset_name)
            results[dataset_name] = success
            
            if success:
                # Estimate review count
                info = self.dataset_catalog[dataset_name]
                size_str = info['size']
                if 'K' in size_str:
                    count = int(size_str.split('K')[0]) * 1000
                elif 'M' in size_str:
                    count = int(float(size_str.split('M')[0]) * 1000000)
                else:
                    count = 0
                total_reviews += count
        
        self.logger.info(f"\n🎉 Dataset Download Summary:")
        self.logger.info(f"   Total Datasets: {sum(results.values())}/{len(priority_datasets)}")
        self.logger.info(f"   Estimated Reviews: {total_reviews:,}")
        self.logger.info(f"   Expected AUC Improvement: +2-5% (0.808 → 0.83-0.85)")
        
        return results
    
    def create_unified_dataset(self, output_file: str = "unified_business_reviews.csv") -> bool:
        """
        Combine all downloaded datasets into a unified format.
        
        Args:
            output_file: Output filename for unified dataset
            
        Returns:
            Success status
        """
        self.logger.info("🔄 Creating unified dataset from all sources...")
        
        unified_data = []
        total_reviews = 0
        
        for dataset_name in self.dataset_catalog.keys():
            dataset_path = self.data_dir / dataset_name
            
            if not dataset_path.exists():
                continue
            
            self.logger.info(f"   Processing: {dataset_name}")
            
            # Load dataset
            csv_files = list(dataset_path.glob("*.csv"))
            if not csv_files:
                continue
            
            try:
                df = pd.read_csv(csv_files[0])
                
                # Standardize column names
                standardized_df = self._standardize_dataset_format(df, dataset_name)
                
                if standardized_df is not None and len(standardized_df) > 0:
                    # Add source information
                    standardized_df['data_source'] = dataset_name
                    standardized_df['source_priority'] = self._get_source_priority(dataset_name)
                    
                    unified_data.append(standardized_df)
                    total_reviews += len(standardized_df)
                    
                    self.logger.info(f"     ✅ Added {len(standardized_df):,} reviews")
                else:
                    self.logger.warning(f"     ⚠️  Skipped {dataset_name} - format issues")
                    
            except Exception as e:
                self.logger.error(f"     ❌ Failed to process {dataset_name}: {str(e)}")
        
        if not unified_data:
            self.logger.error("❌ No valid datasets found to unify")
            return False
        
        # Combine all datasets
        self.logger.info("🔗 Combining datasets...")
        unified_df = pd.concat(unified_data, ignore_index=True)
        
        # Remove duplicates and clean data
        self.logger.info("🧹 Cleaning and deduplicating...")
        unified_df = self._clean_unified_dataset(unified_df)
        
        # Save unified dataset
        output_path = self.data_dir / output_file
        unified_df.to_csv(output_path, index=False)
        
        self.logger.info(f"✅ Unified dataset created:")
        self.logger.info(f"   File: {output_path}")
        self.logger.info(f"   Total Reviews: {len(unified_df):,}")
        self.logger.info(f"   Unique Businesses: {unified_df['business_name'].nunique():,}")
        self.logger.info(f"   Data Sources: {unified_df['data_source'].nunique()}")
        
        return True
    
    def _standardize_dataset_format(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Standardize different dataset formats to common schema."""
        try:
            # Common schema: business_id, business_name, city, state, stars, text, date
            standardized = pd.DataFrame()
            
            # Map columns based on dataset type
            if 'yelp' in dataset_name.lower():
                standardized['business_id'] = df.get('business_id', range(len(df)))
                standardized['business_name'] = df.get('name', df.get('business_name', 'Unknown Business'))
                standardized['city'] = df.get('city', 'Unknown City')
                standardized['state'] = df.get('state', 'Unknown State')
                standardized['stars'] = df.get('stars', df.get('rating', 3.0))
                standardized['text'] = df.get('text', df.get('review_text', ''))
                standardized['date'] = df.get('date', df.get('review_date', '2024-01-01'))
                standardized['categories'] = df.get('categories', 'Business')
                
            elif 'amazon' in dataset_name.lower() or 'food' in dataset_name.lower():
                standardized['business_id'] = df.get('ProductId', range(len(df)))
                standardized['business_name'] = df.get('ProductId', 'Food Business')
                standardized['city'] = 'Various'
                standardized['state'] = 'Various'
                standardized['stars'] = df.get('Score', df.get('rating', 3.0))
                standardized['text'] = df.get('Text', df.get('Summary', ''))
                standardized['date'] = df.get('Time', '2024-01-01')
                standardized['categories'] = 'Food,Restaurant'
                
            elif 'google' in dataset_name.lower():
                standardized['business_id'] = df.get('gmap_id', range(len(df)))
                standardized['business_name'] = df.get('name', 'Google Business')
                standardized['city'] = df.get('address', 'Unknown City')
                standardized['state'] = 'Various'
                standardized['stars'] = df.get('rating', 3.0)
                standardized['text'] = df.get('text', df.get('review_text', ''))
                standardized['date'] = df.get('time', '2024-01-01')
                standardized['categories'] = df.get('category', 'Business')
                
            elif 'tripadvisor' in dataset_name.lower():
                standardized['business_id'] = range(len(df))
                standardized['business_name'] = df.get('Restaurant_Name', 'Restaurant')
                standardized['city'] = df.get('City', 'Unknown City')
                standardized['state'] = 'Various'
                standardized['stars'] = df.get('Rating', 3.0)
                standardized['text'] = df.get('Review', df.get('review_text', ''))
                standardized['date'] = df.get('Date', '2024-01-01')
                standardized['categories'] = 'Restaurant,Food'
            
            else:
                # Generic mapping for unknown formats
                text_cols = [col for col in df.columns if 'text' in col.lower() or 'review' in col.lower()]
                rating_cols = [col for col in df.columns if 'rating' in col.lower() or 'star' in col.lower()]
                name_cols = [col for col in df.columns if 'name' in col.lower()]
                
                standardized['business_id'] = range(len(df))
                standardized['business_name'] = df[name_cols[0]] if name_cols else 'Business'
                standardized['city'] = 'Various'
                standardized['state'] = 'Various'
                standardized['stars'] = df[rating_cols[0]] if rating_cols else 3.0
                standardized['text'] = df[text_cols[0]] if text_cols else ''
                standardized['date'] = '2024-01-01'
                standardized['categories'] = 'Business'
            
            # Clean and validate
            standardized['stars'] = pd.to_numeric(standardized['stars'], errors='coerce').fillna(3.0)
            standardized['text'] = standardized['text'].astype(str).fillna('')
            standardized['business_name'] = standardized['business_name'].astype(str).fillna('Business')
            
            # Filter out rows with empty text
            standardized = standardized[standardized['text'].str.len() > 10]
            
            return standardized
            
        except Exception as e:
            self.logger.error(f"Failed to standardize {dataset_name}: {str(e)}")
            return None
    
    def _clean_unified_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and deduplicate unified dataset."""
        original_size = len(df)
        
        # Remove duplicates based on text similarity
        df = df.drop_duplicates(subset=['business_name', 'text'], keep='first')
        
        # Remove very short or very long reviews
        df = df[(df['text'].str.len() >= 10) & (df['text'].str.len() <= 5000)]
        
        # Remove invalid ratings
        df = df[(df['stars'] >= 1) & (df['stars'] <= 5)]
        
        # Prioritize higher quality sources
        df = df.sort_values(['source_priority', 'stars'], ascending=[True, False])
        
        # Sample to reasonable size (1M reviews max for memory efficiency)
        if len(df) > 1000000:
            df = df.sample(n=1000000, random_state=42)
        
        self.logger.info(f"   Cleaned: {original_size:,} → {len(df):,} reviews")
        
        return df
    
    def _get_source_priority(self, dataset_name: str) -> int:
        """Get priority ranking for data source (lower = higher priority)."""
        priorities = {
            'yelp_review_sentiment': 1,  # Highest quality
            'yelp_reviews_full': 2,
            'google_business_reviews': 3,
            'restaurant_reviews_huge': 4,
            'tripadvisor_restaurant_reviews': 5,
            'business_sentiment_large': 6
        }
        return priorities.get(dataset_name, 10)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Expand Free Review Datasets for Enhanced AI Underwriting"
    )
    parser.add_argument(
        '--download-priority', 
        action='store_true',
        help='Download priority datasets for maximum impact'
    )
    parser.add_argument(
        '--download-all', 
        action='store_true',
        help='Download all available datasets'
    )
    parser.add_argument(
        '--create-unified', 
        action='store_true',
        help='Create unified dataset from downloaded data'
    )
    parser.add_argument(
        '--list-datasets', 
        action='store_true',
        help='List all available datasets'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Download specific dataset by name'
    )
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = ExpandedDatasetManager()
    
    if args.list_datasets:
        manager.list_available_datasets()
        return
    
    if args.download_priority:
        print("🎯 Downloading priority datasets for maximum model improvement...")
        results = manager.download_priority_datasets()
        
        success_count = sum(results.values())
        if success_count > 0:
            print(f"\n✅ Successfully downloaded {success_count} datasets")
            print("🚀 Run with --create-unified to combine datasets")
        else:
            print("\n❌ No datasets downloaded successfully")
            print("💡 Check Kaggle API configuration: https://github.com/Kaggle/kaggle-api")
    
    elif args.download_all:
        print("📥 Downloading all available datasets...")
        results = {}
        for dataset_name in manager.dataset_catalog.keys():
            results[dataset_name] = manager.download_large_dataset(dataset_name)
        
        success_count = sum(results.values())
        print(f"\n✅ Downloaded {success_count}/{len(results)} datasets")
    
    elif args.dataset:
        print(f"📥 Downloading specific dataset: {args.dataset}")
        success = manager.download_large_dataset(args.dataset)
        if success:
            print("✅ Download completed successfully")
        else:
            print("❌ Download failed")
    
    if args.create_unified:
        print("\n🔄 Creating unified dataset...")
        success = manager.create_unified_dataset()
        if success:
            print("✅ Unified dataset created successfully")
            print("🚀 Ready for enhanced model training!")
        else:
            print("❌ Failed to create unified dataset")

if __name__ == "__main__":
    main()