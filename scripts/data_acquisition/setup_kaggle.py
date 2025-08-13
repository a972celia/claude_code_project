#!/usr/bin/env python3
"""
Kaggle API setup script for data acquisition.
"""

import os
import json
import getpass
from pathlib import Path
import logging

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def setup_kaggle_credentials():
    """Set up Kaggle API credentials."""
    logger = logging.getLogger(__name__)
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    credentials_path = kaggle_dir / "kaggle.json"
    
    if credentials_path.exists():
        logger.info("Kaggle credentials already exist.")
        return True
    
    print("Setting up Kaggle API credentials...")
    print("You can find your API credentials at: https://www.kaggle.com/account")
    print("Go to 'Create New API Token' to download kaggle.json")
    
    username = input("Enter your Kaggle username: ")
    key = getpass.getpass("Enter your Kaggle API key: ")
    
    credentials = {
        "username": username,
        "key": key
    }
    
    with open(credentials_path, 'w') as f:
        json.dump(credentials, f)
    
    # Set proper permissions (readable only by user)
    os.chmod(credentials_path, 0o600)
    
    logger.info(f"Kaggle credentials saved to {credentials_path}")
    return True

def verify_kaggle_setup():
    """Verify Kaggle API is working."""
    try:
        import kaggle
        kaggle.api.authenticate()
        
        # Test API connection
        datasets = kaggle.api.dataset_list(search="test", max_size=1)
        print("✅ Kaggle API setup successful!")
        return True
    except Exception as e:
        print(f"❌ Kaggle API setup failed: {str(e)}")
        return False

def main():
    """Main setup function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Kaggle API setup...")
    
    if setup_kaggle_credentials():
        if verify_kaggle_setup():
            logger.info("Kaggle setup completed successfully")
        else:
            logger.error("Kaggle setup verification failed")
    else:
        logger.error("Failed to setup Kaggle credentials")

if __name__ == "__main__":
    main()