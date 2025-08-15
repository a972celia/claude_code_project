"""
Free Business Review Dataset Integration for AI-Powered Underwriting Engine.
Integrates free datasets from Kaggle, Yelp Open Dataset, and other sources for sentiment analysis.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import re
import zipfile
import requests
from dataclasses import dataclass
from datetime import datetime
import pickle
import hashlib

# NLP and Sentiment Analysis
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

@dataclass
class BusinessReviewData:
    """Container for business review information."""
    business_id: str
    business_name: str
    categories: List[str]
    city: str
    state: str
    rating: float
    review_count: int
    reviews: List[Dict]
    sentiment_metrics: Dict[str, float]

class FreeReviewDatasetManager:
    """
    Manager for downloading and processing free business review datasets.
    """
    
    def __init__(self, data_dir: str = "data/external/free_reviews"):
        """
        Initialize the dataset manager.
        
        Args:
            data_dir: Directory for storing downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported datasets
        self.supported_datasets = {
            'yelp_small': {
                'url': 'kaggle_dataset',  # Will use Kaggle API
                'kaggle_ref': 'ilhamfp31/yelp-review-dataset',
                'description': 'Yelp Review Sentiment Dataset (598K reviews)',
                'files': ['yelp_reviews.csv']
            },
            'yelp_polarity': {
                'url': 'kaggle_dataset',
                'kaggle_ref': 'omkarsabnis/yelp-reviews-dataset', 
                'description': 'Yelp Reviews Dataset (650K reviews)',
                'files': ['yelp_reviews.csv']
            },
            'yelp_open': {
                'url': 'https://www.yelp.com/dataset/download',
                'description': 'Official Yelp Open Dataset (4.37GB)',
                'files': [
                    'yelp_academic_dataset_business.json',
                    'yelp_academic_dataset_review.json'
                ]
            }
        }
        
        self.logger = logging.getLogger(__name__)
    
    def list_available_datasets(self) -> Dict[str, str]:
        """List all available free datasets."""
        return {name: info['description'] for name, info in self.supported_datasets.items()}
    
    def download_kaggle_dataset(self, dataset_ref: str, extract_path: Path) -> bool:
        """
        Download dataset from Kaggle using kaggle API.
        
        Args:
            dataset_ref: Kaggle dataset reference (e.g., 'user/dataset-name')
            extract_path: Path to extract files
            
        Returns:
            Success status
        """
        try:
            import kaggle
            
            self.logger.info(f"Downloading Kaggle dataset: {dataset_ref}")
            
            # Download dataset
            kaggle.api.dataset_download_files(
                dataset_ref, 
                path=extract_path, 
                unzip=True
            )
            
            self.logger.info(f"✅ Successfully downloaded: {dataset_ref}")
            return True
            
        except ImportError:
            self.logger.error("❌ Kaggle API not installed. Run: pip install kaggle")
            return False
        except Exception as e:
            self.logger.error(f"❌ Failed to download {dataset_ref}: {str(e)}")
            return False
    
    def download_dataset(self, dataset_name: str) -> bool:
        """
        Download specified dataset.
        
        Args:
            dataset_name: Name of dataset to download
            
        Returns:
            Success status
        """
        if dataset_name not in self.supported_datasets:
            self.logger.error(f"❌ Unknown dataset: {dataset_name}")
            return False
        
        dataset_info = self.supported_datasets[dataset_name]
        dataset_path = self.data_dir / dataset_name
        dataset_path.mkdir(exist_ok=True)
        
        if dataset_info['url'] == 'kaggle_dataset':
            return self.download_kaggle_dataset(
                dataset_info['kaggle_ref'], 
                dataset_path
            )
        else:
            self.logger.info(f"📋 Manual download required for {dataset_name}")
            self.logger.info(f"   URL: {dataset_info['url']}")
            self.logger.info(f"   Extract to: {dataset_path}")
            return False
    
    def check_dataset_availability(self, dataset_name: str) -> bool:
        """Check if dataset is already downloaded."""
        dataset_path = self.data_dir / dataset_name
        if not dataset_path.exists():
            return False
        
        dataset_info = self.supported_datasets[dataset_name]
        required_files = dataset_info.get('files', [])
        
        for file_name in required_files:
            if not (dataset_path / file_name).exists():
                # Check for alternative file names
                alternatives = list(dataset_path.glob(f"*{Path(file_name).suffix}"))
                if not alternatives:
                    return False
        
        return True

class ReviewSentimentAnalyzer:
    """
    Sentiment analysis pipeline for business reviews.
    """
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        self.vectorizer = None
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of individual review text.
        
        Args:
            text: Review text
            
        Returns:
            Sentiment scores dictionary
        """
        try:
            # TextBlob sentiment
            blob = TextBlob(text)
            
            # VADER sentiment (more accurate for social media text)
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            vader = SentimentIntensityAnalyzer()
            vader_scores = vader.polarity_scores(text)
            
            return {
                'textblob_polarity': blob.sentiment.polarity,
                'textblob_subjectivity': blob.sentiment.subjectivity,
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu'], 
                'vader_positive': vader_scores['pos'],
                'vader_compound': vader_scores['compound'],
                'text_length': len(text),
                'word_count': len(text.split())
            }
            
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {str(e)}")
            return {
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 1.0,
                'vader_positive': 0.0,
                'vader_compound': 0.0,
                'text_length': 0,
                'word_count': 0
            }
    
    def analyze_business_reviews(self, reviews: List[Dict]) -> Dict[str, float]:
        """
        Aggregate sentiment analysis for all business reviews.
        
        Args:
            reviews: List of review dictionaries
            
        Returns:
            Aggregated sentiment metrics
        """
        if not reviews:
            return self._empty_sentiment_metrics()
        
        # Analyze each review
        review_sentiments = []
        total_text_length = 0
        total_word_count = 0
        
        for review in reviews:
            text = review.get('text', '')
            if text.strip():
                sentiment = self.analyze_text_sentiment(text)
                review_sentiments.append(sentiment)
                total_text_length += sentiment['text_length']
                total_word_count += sentiment['word_count']
        
        if not review_sentiments:
            return self._empty_sentiment_metrics()
        
        # Aggregate metrics
        df_sentiments = pd.DataFrame(review_sentiments)
        
        aggregated = {
            'avg_textblob_polarity': df_sentiments['textblob_polarity'].mean(),
            'avg_textblob_subjectivity': df_sentiments['textblob_subjectivity'].mean(),
            'avg_vader_compound': df_sentiments['vader_compound'].mean(),
            'avg_vader_positive': df_sentiments['vader_positive'].mean(),
            'avg_vader_negative': df_sentiments['vader_negative'].mean(),
            
            # Variance measures (sentiment consistency)
            'polarity_variance': df_sentiments['textblob_polarity'].var(),
            'vader_variance': df_sentiments['vader_compound'].var(),
            
            # Review engagement metrics
            'review_count_with_text': len(review_sentiments),
            'avg_text_length': total_text_length / len(review_sentiments) if review_sentiments else 0,
            'avg_word_count': total_word_count / len(review_sentiments) if review_sentiments else 0,
            
            # Sentiment distribution
            'positive_review_ratio': (df_sentiments['vader_compound'] > 0.05).mean(),
            'negative_review_ratio': (df_sentiments['vader_compound'] < -0.05).mean(),
            'neutral_review_ratio': (abs(df_sentiments['vader_compound']) <= 0.05).mean(),
        }
        
        return aggregated
    
    def _empty_sentiment_metrics(self) -> Dict[str, float]:
        """Return empty sentiment metrics for businesses without reviews."""
        return {
            'avg_textblob_polarity': 0.0,
            'avg_textblob_subjectivity': 0.0,
            'avg_vader_compound': 0.0,
            'avg_vader_positive': 0.0,
            'avg_vader_negative': 0.0,
            'polarity_variance': 0.0,
            'vader_variance': 0.0,
            'review_count_with_text': 0,
            'avg_text_length': 0.0,
            'avg_word_count': 0.0,
            'positive_review_ratio': 0.0,
            'negative_review_ratio': 0.0,
            'neutral_review_ratio': 0.0
        }

class BusinessNameMatcher:
    """
    Business name matching system to link loan applications with review data.
    """
    
    def __init__(self):
        """Initialize business name matcher."""
        self.business_index = None
        self.logger = logging.getLogger(__name__)
    
    def normalize_business_name(self, name: str) -> str:
        """
        Normalize business name for matching.
        
        Args:
            name: Raw business name
            
        Returns:
            Normalized name
        """
        if not name:
            return ""
        
        # Convert to lowercase
        name = name.lower().strip()
        
        # Remove common business suffixes
        suffixes = [
            'inc', 'llc', 'corp', 'corporation', 'company', 'co', 'ltd',
            'limited', 'enterprises', 'group', 'associates', 'partners'
        ]
        
        for suffix in suffixes:
            # Remove suffix with various punctuation
            patterns = [
                rf'\b{suffix}\b\.?$',
                rf'\b{suffix}\b,?\s*$',
                rf'\s+{suffix}\s*$'
            ]
            for pattern in patterns:
                name = re.sub(pattern, '', name).strip()
        
        # Remove extra whitespace and punctuation
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def build_business_index(self, business_data: List[BusinessReviewData]) -> None:
        """
        Build searchable index of businesses.
        
        Args:
            business_data: List of business review data
        """
        self.business_index = {}
        
        for business in business_data:
            normalized_name = self.normalize_business_name(business.business_name)
            if normalized_name:
                # Store by normalized name
                if normalized_name not in self.business_index:
                    self.business_index[normalized_name] = []
                self.business_index[normalized_name].append(business)
                
                # Also store by business ID
                self.business_index[business.business_id] = business
        
        self.logger.info(f"Built business index with {len(self.business_index)} entries")
    
    def find_business_matches(self, 
                            business_name: str, 
                            location: str = None,
                            threshold: float = 0.8) -> List[BusinessReviewData]:
        """
        Find matching businesses in the review dataset.
        
        Args:
            business_name: Name to search for
            location: Optional location filter
            threshold: Similarity threshold for fuzzy matching
            
        Returns:
            List of matching businesses
        """
        if not self.business_index:
            return []
        
        normalized_query = self.normalize_business_name(business_name)
        matches = []
        
        # Exact match first
        if normalized_query in self.business_index:
            exact_matches = self.business_index[normalized_query]
            if isinstance(exact_matches, list):
                matches.extend(exact_matches)
            else:
                matches.append(exact_matches)
        
        # Fuzzy matching for partial matches
        if not matches:
            from difflib import SequenceMatcher
            
            for indexed_name, businesses in self.business_index.items():
                if isinstance(businesses, list):  # Skip business_id entries
                    similarity = SequenceMatcher(None, normalized_query, indexed_name).ratio()
                    if similarity >= threshold:
                        matches.extend(businesses)
        
        # Filter by location if provided
        if location and matches:
            location_normalized = location.lower()
            location_matches = []
            for business in matches:
                if (location_normalized in business.city.lower() or 
                    location_normalized in business.state.lower()):
                    location_matches.append(business)
            if location_matches:
                matches = location_matches
        
        return matches[:5]  # Return top 5 matches

class FreeReviewDataProcessor:
    """
    Main processor for integrating free review datasets with loan data.
    """
    
    def __init__(self, data_dir: str = "data/external/free_reviews"):
        """
        Initialize the processor.
        
        Args:
            data_dir: Directory containing review datasets
        """
        self.data_manager = FreeReviewDatasetManager(data_dir)
        self.sentiment_analyzer = ReviewSentimentAnalyzer()
        self.name_matcher = BusinessNameMatcher()
        
        self.business_data = []
        self.logger = logging.getLogger(__name__)
    
    def load_yelp_csv_dataset(self, dataset_path: Path) -> List[BusinessReviewData]:
        """
        Load Yelp dataset from CSV format (common Kaggle format).
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            List of business review data
        """
        csv_files = list(dataset_path.glob("*.csv"))
        if not csv_files:
            self.logger.error(f"No CSV files found in {dataset_path}")
            return []
        
        # Load the first CSV file
        csv_file = csv_files[0]
        self.logger.info(f"Loading reviews from: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            self.logger.info(f"Loaded {len(df)} rows from CSV")
            
            # Group reviews by business (if business info available)
            business_data = []
            
            if 'business_id' in df.columns:
                # Group by business_id
                for business_id, business_reviews in df.groupby('business_id'):
                    business_info = business_reviews.iloc[0]
                    
                    reviews = []
                    for _, review in business_reviews.iterrows():
                        reviews.append({
                            'text': review.get('text', ''),
                            'stars': review.get('stars', review.get('rating', 0)),
                            'date': review.get('date', ''),
                            'review_id': review.get('review_id', '')
                        })
                    
                    # Analyze sentiment
                    sentiment_metrics = self.sentiment_analyzer.analyze_business_reviews(reviews)
                    
                    business_data.append(BusinessReviewData(
                        business_id=str(business_id),
                        business_name=business_info.get('name', f'Business_{business_id}'),
                        categories=business_info.get('categories', '').split(',') if business_info.get('categories') else [],
                        city=business_info.get('city', ''),
                        state=business_info.get('state', ''),
                        rating=business_reviews['stars'].mean() if 'stars' in df.columns else 0,
                        review_count=len(business_reviews),
                        reviews=reviews,
                        sentiment_metrics=sentiment_metrics
                    ))
            else:
                # Create synthetic business groupings if no business_id
                # Group by business name or create artificial groupings
                self.logger.warning("No business_id found, creating synthetic business groupings")
                
                # Sample 1000 reviews for demonstration
                sample_df = df.sample(n=min(1000, len(df)), random_state=42)
                
                # Create groups of 10-50 reviews as "businesses"
                group_size = 25
                for i in range(0, len(sample_df), group_size):
                    group = sample_df.iloc[i:i+group_size]
                    
                    reviews = []
                    for _, review in group.iterrows():
                        reviews.append({
                            'text': review.get('text', ''),
                            'stars': review.get('stars', review.get('rating', 0)),
                            'date': review.get('date', ''),
                            'review_id': review.get('review_id', f'review_{i}')
                        })
                    
                    sentiment_metrics = self.sentiment_analyzer.analyze_business_reviews(reviews)
                    
                    business_data.append(BusinessReviewData(
                        business_id=f'synthetic_business_{i//group_size}',
                        business_name=f'Business Group {i//group_size}',
                        categories=['Restaurant'],  # Default category
                        city='City',
                        state='State',
                        rating=group['stars'].mean() if 'stars' in group.columns else 0,
                        review_count=len(group),
                        reviews=reviews,
                        sentiment_metrics=sentiment_metrics
                    ))
            
            self.logger.info(f"Created {len(business_data)} business entries with sentiment analysis")
            return business_data
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV dataset: {str(e)}")
            return []
    
    def load_dataset(self, dataset_name: str) -> bool:
        """
        Load and process a specific dataset.
        
        Args:
            dataset_name: Name of dataset to load
            
        Returns:
            Success status
        """
        if not self.data_manager.check_dataset_availability(dataset_name):
            self.logger.warning(f"Dataset {dataset_name} not found, attempting download...")
            if not self.data_manager.download_dataset(dataset_name):
                return False
        
        dataset_path = self.data_manager.data_dir / dataset_name
        
        # Load based on dataset type
        if dataset_name in ['yelp_small', 'yelp_polarity']:
            self.business_data = self.load_yelp_csv_dataset(dataset_path)
        else:
            self.logger.error(f"Unsupported dataset type: {dataset_name}")
            return False
        
        if self.business_data:
            # Build business name index for matching
            self.name_matcher.build_business_index(self.business_data)
            return True
        
        return False
    
    def enhance_loan_data(self, 
                         loan_df: pd.DataFrame,
                         business_name_col: str = 'business_name',
                         location_col: str = 'business_location') -> pd.DataFrame:
        """
        Enhance loan dataset with free review sentiment data.
        
        Args:
            loan_df: Loan dataset
            business_name_col: Column with business names
            location_col: Column with business locations
            
        Returns:
            Enhanced dataset with review sentiment features
        """
        if not self.business_data:
            self.logger.error("No review data loaded. Load a dataset first.")
            return loan_df
        
        self.logger.info(f"Enhancing {len(loan_df)} loan records with review sentiment data...")
        
        enhanced_features = []
        
        for idx, row in loan_df.iterrows():
            business_name = row.get(business_name_col, '')
            location = row.get(location_col, '')
            
            if business_name:
                # Find matching businesses
                matches = self.name_matcher.find_business_matches(business_name, location)
                
                if matches:
                    # Use the best match
                    best_match = matches[0]
                    features = self._extract_review_features(best_match)
                    features['review_data_source'] = 'matched'
                    features['review_match_confidence'] = 1.0 / len(matches)  # Higher confidence for fewer matches
                else:
                    features = self._empty_review_features()
                    features['review_data_source'] = 'not_found'
                    features['review_match_confidence'] = 0.0
            else:
                features = self._empty_review_features()
                features['review_data_source'] = 'no_business_name'
                features['review_match_confidence'] = 0.0
            
            enhanced_features.append(features)
        
        # Convert to DataFrame and merge
        features_df = pd.DataFrame(enhanced_features)
        enhanced_df = pd.concat([loan_df.reset_index(drop=True), features_df], axis=1)
        
        self.logger.info(f"✅ Enhanced dataset with {len(features_df.columns)} review features")
        return enhanced_df
    
    def _extract_review_features(self, business: BusinessReviewData) -> Dict[str, float]:
        """Extract ML features from business review data."""
        sentiment = business.sentiment_metrics
        
        features = {
            # Basic business metrics
            'review_rating': business.rating,
            'review_count': business.review_count,
            'review_categories_count': len(business.categories),
            
            # Sentiment features
            'review_avg_polarity': sentiment.get('avg_textblob_polarity', 0.0),
            'review_avg_subjectivity': sentiment.get('avg_textblob_subjectivity', 0.0),
            'review_vader_compound': sentiment.get('avg_vader_compound', 0.0),
            'review_vader_positive': sentiment.get('avg_vader_positive', 0.0),
            'review_vader_negative': sentiment.get('avg_vader_negative', 0.0),
            
            # Sentiment consistency
            'review_polarity_variance': sentiment.get('polarity_variance', 0.0),
            'review_vader_variance': sentiment.get('vader_variance', 0.0),
            
            # Review engagement
            'review_text_count': sentiment.get('review_count_with_text', 0),
            'review_avg_length': sentiment.get('avg_text_length', 0.0),
            'review_avg_words': sentiment.get('avg_word_count', 0.0),
            
            # Sentiment distribution
            'review_positive_ratio': sentiment.get('positive_review_ratio', 0.0),
            'review_negative_ratio': sentiment.get('negative_review_ratio', 0.0),
            'review_neutral_ratio': sentiment.get('neutral_review_ratio', 0.0),
            
            # Risk indicators
            'review_poor_rating': 1 if business.rating < 3.0 else 0,
            'review_negative_sentiment': 1 if sentiment.get('avg_vader_compound', 0.0) < -0.1 else 0,
            'review_low_engagement': 1 if sentiment.get('review_count_with_text', 0) < 5 else 0,
            
            # Composite scores
            'review_reputation_score': self._calculate_reputation_score(business, sentiment),
            'review_consistency_score': self._calculate_consistency_score(sentiment)
        }
        
        return features
    
    def _empty_review_features(self) -> Dict[str, float]:
        """Return empty features for businesses without review data."""
        return {
            'review_rating': 0.0,
            'review_count': 0,
            'review_categories_count': 0,
            'review_avg_polarity': 0.0,
            'review_avg_subjectivity': 0.0,
            'review_vader_compound': 0.0,
            'review_vader_positive': 0.0,
            'review_vader_negative': 0.0,
            'review_polarity_variance': 0.0,
            'review_vader_variance': 0.0,
            'review_text_count': 0,
            'review_avg_length': 0.0,
            'review_avg_words': 0.0,
            'review_positive_ratio': 0.0,
            'review_negative_ratio': 0.0,
            'review_neutral_ratio': 0.0,
            'review_poor_rating': 0,
            'review_negative_sentiment': 0,
            'review_low_engagement': 0,
            'review_reputation_score': 0.0,
            'review_consistency_score': 0.0
        }
    
    def _calculate_reputation_score(self, business: BusinessReviewData, sentiment: Dict) -> float:
        """Calculate composite reputation score."""
        rating_component = business.rating / 5.0 if business.rating > 0 else 0
        sentiment_component = (sentiment.get('avg_vader_compound', 0.0) + 1) / 2  # Normalize to 0-1
        
        # Weight by review count (more reviews = more reliable)
        review_weight = min(business.review_count / 50, 1.0)
        
        return (rating_component * 0.6 + sentiment_component * 0.4) * review_weight
    
    def _calculate_consistency_score(self, sentiment: Dict) -> float:
        """Calculate sentiment consistency score (lower variance = higher consistency)."""
        polarity_var = sentiment.get('polarity_variance', 0.0)
        vader_var = sentiment.get('vader_variance', 0.0)
        
        # Convert variance to consistency score (inverse relationship)
        avg_variance = (polarity_var + vader_var) / 2
        consistency = max(0, 1 - avg_variance)  # Higher variance = lower consistency
        
        return consistency