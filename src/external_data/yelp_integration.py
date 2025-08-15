"""
Yelp Business Data Integration for AI-Powered Underwriting Engine.
Integrates Yelp business data, reviews, and sentiment analysis for enhanced risk assessment.
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import re
from dataclasses import dataclass
import hashlib

# Sentiment analysis
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

@dataclass
class YelpBusinessData:
    """Container for Yelp business data."""
    id: str
    name: str
    rating: float
    review_count: int
    categories: List[str]
    price: str
    location: Dict
    phone: str
    display_phone: str
    url: str
    is_closed: bool
    coordinates: Dict
    photos: List[str]
    hours: Optional[Dict]
    reviews: List[Dict]
    sentiment_scores: Dict

class YelpDataCollector:
    """
    Collect and process Yelp business data for underwriting enhancement.
    """
    
    def __init__(self, api_key: str, cache_dir: str = "data/external/yelp"):
        """
        Initialize Yelp data collector.
        
        Args:
            api_key: Yelp Fusion API key
            cache_dir: Directory for caching API responses
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_url = "https://api.yelp.com/v3"
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests per second max
        
        self.logger = logging.getLogger(__name__)
        
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """
        Make rate-limited API request with caching.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            API response data or None if failed
        """
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        # Check cache first
        cache_key = hashlib.md5(f"{endpoint}{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 86400:  # Cache for 24 hours
                self.logger.debug(f"Using cached data for {endpoint}")
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        try:
            url = f"{self.base_url}/{endpoint}"
            self.logger.info(f"Making request to {endpoint} with params: {params}")
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            self.last_request_time = time.time()
            data = response.json()
            
            # Cache response
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return data
            
        except requests.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            return None
    
    def search_businesses(self, 
                         location: str = None,
                         latitude: float = None, 
                         longitude: float = None,
                         categories: str = None,
                         radius: int = 10000,
                         limit: int = 50) -> List[Dict]:
        """
        Search for businesses using Yelp API.
        
        Args:
            location: Location string (e.g., "New York, NY")
            latitude: Latitude coordinate
            longitude: Longitude coordinate  
            categories: Business categories (e.g., "restaurants,bars")
            radius: Search radius in meters (max 40000)
            limit: Number of results to return (max 50 per request)
            
        Returns:
            List of business data dictionaries
        """
        params = {
            'limit': min(limit, 50),
            'radius': min(radius, 40000),
        }
        
        if location:
            params['location'] = location
        elif latitude and longitude:
            params['latitude'] = latitude
            params['longitude'] = longitude
        else:
            raise ValueError("Must provide either location or coordinates")
        
        if categories:
            params['categories'] = categories
            
        data = self._make_request('businesses/search', params)
        return data.get('businesses', []) if data else []
    
    def get_business_details(self, business_id: str) -> Optional[Dict]:
        """
        Get detailed business information.
        
        Args:
            business_id: Yelp business ID
            
        Returns:
            Business details dictionary
        """
        return self._make_request(f'businesses/{business_id}', {})
    
    def get_business_reviews(self, business_id: str) -> List[Dict]:
        """
        Get business reviews (limited to 3 excerpts by Yelp API).
        
        Args:
            business_id: Yelp business ID
            
        Returns:
            List of review dictionaries
        """
        data = self._make_request(f'businesses/{business_id}/reviews', {})
        return data.get('reviews', []) if data else []
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of review text using TextBlob.
        
        Args:
            text: Review text
            
        Returns:
            Sentiment scores dictionary
        """
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,      # -1 (negative) to 1 (positive)
                'subjectivity': blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
            }
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {str(e)}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def collect_business_data(self, business_id: str) -> Optional[YelpBusinessData]:
        """
        Collect comprehensive business data including reviews and sentiment.
        
        Args:
            business_id: Yelp business ID
            
        Returns:
            YelpBusinessData object or None
        """
        # Get business details
        business = self.get_business_details(business_id)
        if not business:
            return None
        
        # Get reviews
        reviews = self.get_business_reviews(business_id)
        
        # Analyze sentiment for each review
        sentiment_scores = []
        for review in reviews:
            sentiment = self.analyze_sentiment(review.get('text', ''))
            sentiment_scores.append(sentiment)
        
        # Aggregate sentiment metrics
        if sentiment_scores:
            avg_polarity = np.mean([s['polarity'] for s in sentiment_scores])
            avg_subjectivity = np.mean([s['subjectivity'] for s in sentiment_scores])
            sentiment_variance = np.var([s['polarity'] for s in sentiment_scores])
        else:
            avg_polarity = avg_subjectivity = sentiment_variance = 0.0
        
        aggregated_sentiment = {
            'avg_polarity': avg_polarity,
            'avg_subjectivity': avg_subjectivity,
            'sentiment_variance': sentiment_variance,
            'review_count_with_text': len([r for r in reviews if r.get('text')])
        }
        
        # Extract categories
        categories = [cat['title'] for cat in business.get('categories', [])]
        
        return YelpBusinessData(
            id=business.get('id', ''),
            name=business.get('name', ''),
            rating=business.get('rating', 0.0),
            review_count=business.get('review_count', 0),
            categories=categories,
            price=business.get('price', ''),
            location=business.get('location', {}),
            phone=business.get('phone', ''),
            display_phone=business.get('display_phone', ''),
            url=business.get('url', ''),
            is_closed=business.get('is_closed', False),
            coordinates=business.get('coordinates', {}),
            photos=business.get('photos', []),
            hours=business.get('hours'),
            reviews=reviews,
            sentiment_scores=aggregated_sentiment
        )

class YelpFeatureEngineer:
    """
    Engineer features from Yelp data for underwriting models.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_business_features(self, yelp_data: YelpBusinessData) -> Dict[str, Union[float, int, str]]:
        """
        Create features from Yelp business data.
        
        Args:
            yelp_data: YelpBusinessData object
            
        Returns:
            Dictionary of engineered features
        """
        features = {}
        
        # Basic business metrics
        features['yelp_rating'] = yelp_data.rating
        features['yelp_review_count'] = yelp_data.review_count
        features['yelp_is_closed'] = int(yelp_data.is_closed)
        
        # Price level (convert $ symbols to numeric)
        price_map = {'': 0, '$': 1, '$$': 2, '$$$': 3, '$$$$': 4}
        features['yelp_price_level'] = price_map.get(yelp_data.price, 0)
        
        # Business longevity proxy (more reviews often = longer established)
        features['yelp_establishment_score'] = min(yelp_data.review_count / 100, 10.0)
        
        # Category analysis
        features['yelp_category_count'] = len(yelp_data.categories)
        
        # High-risk categories (restaurants, bars have higher failure rates)
        high_risk_categories = ['restaurants', 'bars', 'food', 'nightlife']
        features['yelp_high_risk_category'] = int(any(
            cat.lower() in high_risk_categories 
            for cat in yelp_data.categories
        ))
        
        # Sentiment features
        sentiment = yelp_data.sentiment_scores
        features['yelp_sentiment_polarity'] = sentiment.get('avg_polarity', 0.0)
        features['yelp_sentiment_subjectivity'] = sentiment.get('avg_subjectivity', 0.0)
        features['yelp_sentiment_variance'] = sentiment.get('sentiment_variance', 0.0)
        features['yelp_reviews_with_text'] = sentiment.get('review_count_with_text', 0)
        
        # Reputation risk score (combination of rating and sentiment)
        if yelp_data.rating > 0:
            reputation_score = (
                yelp_data.rating / 5.0 * 0.6 +  # Rating weight
                (sentiment.get('avg_polarity', 0.0) + 1) / 2 * 0.4  # Sentiment weight
            )
            features['yelp_reputation_score'] = reputation_score
            
            # Risk indicators
            features['yelp_poor_rating'] = int(yelp_data.rating < 3.0)
            features['yelp_negative_sentiment'] = int(sentiment.get('avg_polarity', 0.0) < -0.1)
            
            # Engagement metrics
            if yelp_data.review_count > 0:
                features['yelp_engagement_ratio'] = min(
                    sentiment.get('review_count_with_text', 0) / yelp_data.review_count, 1.0
                )
            else:
                features['yelp_engagement_ratio'] = 0.0
        else:
            # No Yelp presence (potential risk factor)
            features['yelp_reputation_score'] = 0.0
            features['yelp_poor_rating'] = 0
            features['yelp_negative_sentiment'] = 0
            features['yelp_engagement_ratio'] = 0.0
        
        # Photo engagement (more photos often indicate active business)
        features['yelp_photo_count'] = len(yelp_data.photos)
        features['yelp_has_photos'] = int(len(yelp_data.photos) > 0)
        
        return features
    
    def batch_create_features(self, yelp_data_list: List[YelpBusinessData]) -> pd.DataFrame:
        """
        Create features for multiple businesses.
        
        Args:
            yelp_data_list: List of YelpBusinessData objects
            
        Returns:
            DataFrame with engineered features
        """
        feature_dicts = []
        
        for yelp_data in yelp_data_list:
            features = self.create_business_features(yelp_data)
            features['yelp_business_id'] = yelp_data.id
            features['yelp_business_name'] = yelp_data.name
            feature_dicts.append(features)
        
        return pd.DataFrame(feature_dicts)

class YelpDataPipeline:
    """
    Complete pipeline for integrating Yelp data with loan underwriting.
    """
    
    def __init__(self, api_key: str, cache_dir: str = "data/external/yelp"):
        """
        Initialize Yelp data pipeline.
        
        Args:
            api_key: Yelp Fusion API key
            cache_dir: Directory for caching data
        """
        self.collector = YelpDataCollector(api_key, cache_dir)
        self.feature_engineer = YelpFeatureEngineer()
        self.logger = logging.getLogger(__name__)
    
    def enhance_loan_data(self, 
                         loan_df: pd.DataFrame,
                         business_name_col: str = 'business_name',
                         location_col: str = 'business_location') -> pd.DataFrame:
        """
        Enhance loan dataset with Yelp business data.
        
        Args:
            loan_df: Loan dataset DataFrame
            business_name_col: Column containing business names
            location_col: Column containing business locations
            
        Returns:
            Enhanced DataFrame with Yelp features
        """
        self.logger.info(f"Enhancing {len(loan_df)} loan records with Yelp data")
        
        yelp_features_list = []
        
        for idx, row in loan_df.iterrows():
            business_name = row.get(business_name_col, '')
            location = row.get(location_col, '')
            
            if not business_name or not location:
                # Create empty features for missing business info
                empty_features = self._create_empty_features()
                yelp_features_list.append(empty_features)
                continue
            
            try:
                # Search for business
                businesses = self.collector.search_businesses(
                    location=location,
                    term=business_name,
                    limit=1
                )
                
                if businesses:
                    business_id = businesses[0]['id']
                    yelp_data = self.collector.collect_business_data(business_id)
                    
                    if yelp_data:
                        features = self.feature_engineer.create_business_features(yelp_data)
                    else:
                        features = self._create_empty_features()
                else:
                    # Business not found on Yelp
                    features = self._create_empty_features()
                    features['yelp_not_found'] = 1
                
                yelp_features_list.append(features)
                
            except Exception as e:
                self.logger.warning(f"Failed to process business {business_name}: {str(e)}")
                features = self._create_empty_features()
                features['yelp_error'] = 1
                yelp_features_list.append(features)
        
        # Convert to DataFrame and merge
        yelp_features_df = pd.DataFrame(yelp_features_list)
        enhanced_df = pd.concat([loan_df.reset_index(drop=True), yelp_features_df], axis=1)
        
        self.logger.info(f"Successfully enhanced dataset with {len(yelp_features_df.columns)} Yelp features")
        return enhanced_df
    
    def _create_empty_features(self) -> Dict[str, Union[float, int]]:
        """Create empty feature set for businesses without Yelp data."""
        return {
            'yelp_rating': 0.0,
            'yelp_review_count': 0,
            'yelp_is_closed': 0,
            'yelp_price_level': 0,
            'yelp_establishment_score': 0.0,
            'yelp_category_count': 0,
            'yelp_high_risk_category': 0,
            'yelp_sentiment_polarity': 0.0,
            'yelp_sentiment_subjectivity': 0.0,
            'yelp_sentiment_variance': 0.0,
            'yelp_reviews_with_text': 0,
            'yelp_reputation_score': 0.0,
            'yelp_poor_rating': 0,
            'yelp_negative_sentiment': 0,
            'yelp_engagement_ratio': 0.0,
            'yelp_photo_count': 0,
            'yelp_has_photos': 0,
            'yelp_not_found': 0,
            'yelp_error': 0
        }