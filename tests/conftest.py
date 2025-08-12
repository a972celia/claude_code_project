"""
Pytest configuration and shared fixtures.
"""

import pytest
import os
from pathlib import Path

@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_business_data():
    """Sample business data for testing."""
    return {
        "business_id": "test-business-123",
        "business_name": "Test Restaurant",
        "industry": "Food Service",
        "location": "San Francisco, CA",
        "years_in_business": 3,
        "annual_revenue": 500000.0
    }