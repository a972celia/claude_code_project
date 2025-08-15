"""
Yelp API Configuration for AI-Powered Underwriting Engine.
Manages API credentials and settings for external data integration.
"""

import os
from pathlib import Path
from typing import Optional
import logging

class YelpConfig:
    """Configuration manager for Yelp API integration."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Try to load from environment variables first
        self.api_key = os.getenv('YELP_API_KEY')
        
        # If not in environment, try to load from config file
        if not self.api_key:
            self.api_key = self._load_from_config_file()
        
        # API settings
        self.base_url = "https://api.yelp.com/v3"
        self.rate_limit = 0.1  # Minimum seconds between requests
        self.cache_duration = 86400  # Cache responses for 24 hours
        
        # Data collection settings
        self.default_search_radius = 10000  # 10km
        self.max_businesses_per_search = 50
        self.max_businesses_total = 240  # Yelp API limit
        
        # Feature engineering settings
        self.high_risk_categories = [
            'restaurants', 'bars', 'food', 'nightlife', 'coffee', 'cafes'
        ]
        
        # Sentiment analysis thresholds
        self.negative_sentiment_threshold = -0.1
        self.positive_sentiment_threshold = 0.1
        self.poor_rating_threshold = 3.0
        
    def _load_from_config_file(self) -> Optional[str]:
        """Load API key from configuration file."""
        config_file = Path("config/yelp_api_key.txt")
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    api_key = f.read().strip()
                    if api_key:
                        return api_key
            except Exception as e:
                self.logger.warning(f"Failed to read config file: {str(e)}")
        
        return None
    
    def is_configured(self) -> bool:
        """Check if Yelp API is properly configured."""
        return self.api_key is not None
    
    def get_api_key(self) -> str:
        """Get Yelp API key with validation."""
        if not self.api_key:
            raise ValueError(
                "Yelp API key not configured. Please set YELP_API_KEY environment "
                "variable or create config/yelp_api_key.txt file."
            )
        return self.api_key
    
    def setup_instructions(self) -> str:
        """Return setup instructions for Yelp API."""
        return """
🔑 YELP API SETUP INSTRUCTIONS

To use Yelp data integration, you need to obtain a Yelp Fusion API key:

1. Visit: https://docs.developer.yelp.com/docs/fusion-intro
2. Sign up for a Yelp developer account
3. Create a new app to get your API key
4. Configure your API key using ONE of these methods:

   Method 1 - Environment Variable (Recommended):
   export YELP_API_KEY="your_api_key_here"
   
   Method 2 - Configuration File:
   Create: config/yelp_api_key.txt
   Content: your_api_key_here

⚠️  IMPORTANT: Never commit your API key to version control!
   Add config/yelp_api_key.txt to .gitignore

📋 API Limits:
- Rate Limit: 5000 requests/day
- Search Results: Up to 240 businesses per search
- Reviews: 3 excerpt reviews per business
- Geographic Coverage: Businesses with user-generated content

💡 For production use, consider Yelp Fusion Insights for higher limits
   and full review access.
        """.strip()

# Global configuration instance
yelp_config = YelpConfig()