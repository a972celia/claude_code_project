"""
Enhanced Data Preprocessor with External Data Integration.
Extends the base preprocessor to include Yelp and other external data sources.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from data_pipeline.preprocessing.data_preprocessor import LoanDataPreprocessor
from external_data.yelp_integration import YelpDataPipeline
from config.yelp_config import yelp_config

class EnhancedLoanDataPreprocessor(LoanDataPreprocessor):
    """
    Enhanced preprocessor that integrates external data sources like Yelp
    with the existing loan preprocessing pipeline.
    """
    
    def __init__(self,
                 missing_threshold: float = 0.6,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42,
                 enable_yelp: bool = True):
        """
        Initialize enhanced preprocessor.
        
        Args:
            missing_threshold: Threshold for dropping features with missing values
            test_size: Test set proportion
            val_size: Validation set proportion  
            random_state: Random seed
            enable_yelp: Whether to enable Yelp data integration
        """
        super().__init__(missing_threshold, test_size, val_size, random_state)
        
        self.enable_yelp = enable_yelp
        self.yelp_pipeline = None
        
        if self.enable_yelp and yelp_config.is_configured():
            try:
                api_key = yelp_config.get_api_key()
                self.yelp_pipeline = YelpDataPipeline(api_key)
                self.logger.info("✅ Yelp integration enabled")
            except Exception as e:
                self.logger.warning(f"⚠️ Yelp integration disabled: {str(e)}")
                self.enable_yelp = False
        elif self.enable_yelp:
            self.logger.warning("⚠️ Yelp integration disabled: API key not configured")
            self.enable_yelp = False
        
        # Extended feature lists for external data
        self.yelp_features = [
            'yelp_rating', 'yelp_review_count', 'yelp_is_closed', 'yelp_price_level',
            'yelp_establishment_score', 'yelp_category_count', 'yelp_high_risk_category',
            'yelp_sentiment_polarity', 'yelp_sentiment_subjectivity', 'yelp_sentiment_variance',
            'yelp_reviews_with_text', 'yelp_reputation_score', 'yelp_poor_rating',
            'yelp_negative_sentiment', 'yelp_engagement_ratio', 'yelp_photo_count',
            'yelp_has_photos', 'yelp_not_found', 'yelp_error'
        ]
    
    def integrate_external_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate external data sources into the loan dataset.
        
        Args:
            df: Input loan dataset
            
        Returns:
            Dataset enhanced with external data
        """
        self.logger.info("🌐 Integrating external data sources...")
        enhanced_df = df.copy()
        
        if self.enable_yelp and self.yelp_pipeline:
            enhanced_df = self._integrate_yelp_data(enhanced_df)
        
        return enhanced_df
    
    def _integrate_yelp_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate Yelp business data.
        
        Args:
            df: Input dataset
            
        Returns:
            Dataset with Yelp features
        """
        self.logger.info("🟡 Integrating Yelp business data...")
        
        # Check if we have business information in the dataset
        business_cols = self._identify_business_columns(df)
        
        if not business_cols['name'] and not business_cols['location']:
            self.logger.warning("⚠️ No business information found for Yelp integration")
            # Add empty Yelp features
            return self._add_empty_yelp_features(df)
        
        try:
            # For demonstration, we'll create synthetic business data
            # In a real scenario, you'd extract from loan applications
            enhanced_df = self._create_synthetic_yelp_features(df)
            
            self.logger.info(f"✅ Successfully integrated Yelp features")
            return enhanced_df
            
        except Exception as e:
            self.logger.error(f"❌ Yelp integration failed: {str(e)}")
            return self._add_empty_yelp_features(df)
    
    def _identify_business_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        Identify columns containing business information.
        
        Args:
            df: Input dataset
            
        Returns:
            Dictionary with identified column names
        """
        name_candidates = ['business_name', 'company_name', 'name', 'BusinessName']
        location_candidates = ['business_location', 'location', 'address', 'city', 'BusinessLocation']
        
        business_cols = {'name': None, 'location': None}
        
        for col in df.columns:
            if col.lower() in [c.lower() for c in name_candidates]:
                business_cols['name'] = col
            elif col.lower() in [c.lower() for c in location_candidates]:
                business_cols['location'] = col
        
        return business_cols
    
    def _create_synthetic_yelp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic Yelp features for demonstration.
        In production, this would call the actual Yelp API.
        
        Args:
            df: Input dataset
            
        Returns:
            Dataset with synthetic Yelp features
        """
        self.logger.info("🔧 Creating synthetic Yelp features for demonstration...")
        
        np.random.seed(self.random_state)
        n_samples = len(df)
        
        # Create realistic synthetic Yelp features based on loan characteristics
        # Higher risk loans tend to have businesses with poorer Yelp metrics
        
        # Use Interest rate as a proxy for loan risk to create correlated Yelp features
        if 'Interest' in df.columns:
            risk_proxy = (df['Interest'] - df['Interest'].min()) / (df['Interest'].max() - df['Interest'].min())
        else:
            risk_proxy = np.random.random(n_samples)
        
        synthetic_features = {
            # Basic Yelp metrics (inversely correlated with risk)
            'yelp_rating': np.clip(4.5 - risk_proxy * 2 + np.random.normal(0, 0.3, n_samples), 1.0, 5.0),
            'yelp_review_count': np.random.poisson(50 + (1 - risk_proxy) * 100, n_samples),
            'yelp_is_closed': np.random.binomial(1, risk_proxy * 0.1, n_samples),
            'yelp_price_level': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
            
            # Establishment and reputation metrics
            'yelp_establishment_score': np.clip((1 - risk_proxy) * 5 + np.random.normal(0, 0.5, n_samples), 0, 10),
            'yelp_category_count': np.random.poisson(2, n_samples) + 1,
            'yelp_high_risk_category': np.random.binomial(1, 0.3, n_samples),
            
            # Sentiment features (correlated with rating)
            'yelp_sentiment_polarity': np.clip(np.random.normal(0.1, 0.2, n_samples), -1, 1),
            'yelp_sentiment_subjectivity': np.random.uniform(0.3, 0.8, n_samples),
            'yelp_sentiment_variance': np.random.exponential(0.1, n_samples),
            'yelp_reviews_with_text': np.random.binomial(10, 0.7, n_samples),
            
            # Composite scores
            'yelp_reputation_score': 0.0,  # Will be calculated below
            'yelp_poor_rating': 0,         # Will be calculated below
            'yelp_negative_sentiment': 0,  # Will be calculated below
            
            # Engagement metrics
            'yelp_engagement_ratio': np.random.uniform(0.5, 1.0, n_samples),
            'yelp_photo_count': np.random.poisson(5, n_samples),
            'yelp_has_photos': np.random.binomial(1, 0.8, n_samples),
            
            # Data availability flags
            'yelp_not_found': np.random.binomial(1, 0.05, n_samples),
            'yelp_error': np.random.binomial(1, 0.02, n_samples)
        }
        
        # Calculate composite features
        synthetic_features['yelp_reputation_score'] = (
            synthetic_features['yelp_rating'] / 5.0 * 0.7 +
            (synthetic_features['yelp_sentiment_polarity'] + 1) / 2 * 0.3
        )
        
        synthetic_features['yelp_poor_rating'] = (synthetic_features['yelp_rating'] < 3.0).astype(int)
        synthetic_features['yelp_negative_sentiment'] = (synthetic_features['yelp_sentiment_polarity'] < -0.1).astype(int)
        
        # Add synthetic features to dataset
        enhanced_df = df.copy()
        for feature, values in synthetic_features.items():
            enhanced_df[feature] = values
        
        return enhanced_df
    
    def _add_empty_yelp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add empty Yelp features when integration is not available.
        
        Args:
            df: Input dataset
            
        Returns:
            Dataset with zero-filled Yelp features
        """
        enhanced_df = df.copy()
        
        for feature in self.yelp_features:
            enhanced_df[feature] = 0
        
        return enhanced_df
    
    def preprocess_with_external_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline with external data integration.
        
        Args:
            df: Raw loan dataset
            
        Returns:
            Preprocessed X_train, X_val, X_test, y_train, y_val, y_test
        """
        self.logger.info("🚀 Starting enhanced preprocessing with external data...")
        
        # Step 1: Integrate external data sources
        enhanced_df = self.integrate_external_data(df)
        
        # Step 2: Run standard preprocessing pipeline
        X_train, X_val, X_test, y_train, y_val, y_test = self.fit_transform(enhanced_df)
        
        # Update feature lists to include external features
        if self.enable_yelp:
            # Add Yelp features to numeric features for scaling
            available_yelp_features = [f for f in self.yelp_features if f in X_train.columns]
            self.numeric_features.extend(available_yelp_features)
            
            self.logger.info(f"✅ Added {len(available_yelp_features)} Yelp features to the model")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def generate_enhanced_summary(self) -> str:
        """
        Generate preprocessing summary including external data integration.
        
        Returns:
            Enhanced summary string
        """
        # Generate base summary from feature summary
        feature_summary = self.get_feature_summary()
        base_summary = f"""PREPROCESSING PIPELINE SUMMARY
==================================================

features_dropped: {feature_summary.get('features_dropped', 0)}
features_selected: {feature_summary.get('features_selected', 0)}
numeric_features: {feature_summary.get('numeric_features', 0)}
categorical_features: {feature_summary.get('categorical_features', 0)}
engineered_features: {feature_summary.get('engineered_features', 0)}
missing_threshold: {self.missing_threshold}
preprocessing_steps:
  - Mixed data types handling
  - Target variable creation
  - Feature dropping
  - Outlier handling
  - Feature engineering
  - Feature selection
  - Missing value imputation
  - Categorical encoding
  - Numeric scaling
  - External data integration

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        external_summary = "\n\nEXTERNAL DATA INTEGRATION\n"
        external_summary += "=" * 50 + "\n"
        
        if self.enable_yelp:
            external_summary += f"✅ Yelp Integration: ENABLED\n"
            external_summary += f"   - Features Added: {len(self.yelp_features)}\n"
            external_summary += f"   - API Status: {'Configured' if yelp_config.is_configured() else 'Not Configured'}\n"
        else:
            external_summary += f"❌ Yelp Integration: DISABLED\n"
        
        external_summary += f"\nTotal Enhanced Features: {len(self.numeric_features) + len(self.categorical_features)}\n"
        
        return base_summary + external_summary