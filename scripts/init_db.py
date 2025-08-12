#!/usr/bin/env python3
"""
Database initialization script for the AI-Powered Underwriting Engine.
"""

import os
import sys
from pathlib import Path
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_environment():
    """Load environment variables."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logging.warning("No .env file found. Using environment variables.")

def create_database():
    """Create database and tables."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    engine = create_engine(database_url)
    
    # Create tables (placeholder - implement actual schema)
    with engine.connect() as conn:
        # Business data table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS businesses (
                id SERIAL PRIMARY KEY,
                business_id VARCHAR(255) UNIQUE NOT NULL,
                name VARCHAR(255) NOT NULL,
                industry VARCHAR(100),
                location VARCHAR(255),
                years_in_business INTEGER,
                annual_revenue DECIMAL(15,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # Risk assessments table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS risk_assessments (
                id SERIAL PRIMARY KEY,
                business_id VARCHAR(255) REFERENCES businesses(business_id),
                risk_score DECIMAL(5,4),
                risk_category VARCHAR(50),
                model_version VARCHAR(20),
                assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                key_factors JSONB
            )
        """))
        
        # Model performance tracking
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id SERIAL PRIMARY KEY,
                model_version VARCHAR(20),
                metric_name VARCHAR(50),
                metric_value DECIMAL(10,6),
                evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        conn.commit()
    
    logging.info("Database tables created successfully")

def main():
    """Main initialization function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting database initialization...")
    
    try:
        load_environment()
        create_database()
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()