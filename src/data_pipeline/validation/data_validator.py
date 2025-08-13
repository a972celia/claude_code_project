"""
Data validation utilities for the AI-Powered Underwriting Engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json

class DataValidator:
    """Class for validating data quality and integrity."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
    
    def validate_dataframe(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Comprehensive validation of a pandas DataFrame.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name identifier for the dataset
            
        Returns:
            Dictionary containing validation results
        """
        self.logger.info(f"Validating dataset: {dataset_name}")
        
        results = {
            "dataset_name": dataset_name,
            "validation_timestamp": pd.Timestamp.now().isoformat(),
            "basic_info": self._validate_basic_info(df),
            "missing_data": self._validate_missing_data(df),
            "data_types": self._validate_data_types(df),
            "duplicates": self._validate_duplicates(df),
            "outliers": self._validate_outliers(df),
            "data_quality_score": 0.0
        }
        
        # Calculate overall data quality score
        results["data_quality_score"] = self._calculate_quality_score(results)
        
        self.validation_results[dataset_name] = results
        return results
    
    def _validate_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate basic DataFrame information."""
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "column_names": list(df.columns)
        }
    
    def _validate_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate missing data patterns."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            "total_missing_values": int(missing_counts.sum()),
            "columns_with_missing": missing_counts[missing_counts > 0].to_dict(),
            "missing_percentages": missing_percentages[missing_percentages > 0].to_dict(),
            "completely_empty_columns": missing_counts[missing_counts == len(df)].index.tolist()
        }
    
    def _validate_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data types and suggest improvements."""
        dtypes_info = df.dtypes.astype(str).to_dict()
        
        # Identify potential type issues
        potential_issues = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric columns are stored as strings
                try:
                    pd.to_numeric(df[col].dropna().head(100))
                    potential_issues.append({
                        "column": col,
                        "issue": "numeric_as_string",
                        "suggestion": "Convert to numeric type"
                    })
                except:
                    pass
                
                # Check for date columns
                if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                    potential_issues.append({
                        "column": col,
                        "issue": "potential_datetime",
                        "suggestion": "Consider converting to datetime"
                    })
        
        return {
            "data_types": dtypes_info,
            "potential_issues": potential_issues
        }
    
    def _validate_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate duplicate records."""
        total_duplicates = df.duplicated().sum()
        
        return {
            "total_duplicate_rows": int(total_duplicates),
            "duplicate_percentage": round((total_duplicates / len(df)) * 100, 2),
            "unique_rows": len(df) - total_duplicates
        }
    
    def _validate_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate outliers in numeric columns."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_columns:
            if df[col].notna().sum() > 0:  # Only process if there are non-null values
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                
                outlier_info[col] = {
                    "outlier_count": len(outliers),
                    "outlier_percentage": round((len(outliers) / len(df)) * 100, 2),
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "min_value": float(df[col].min()),
                    "max_value": float(df[col].max())
                }
        
        return outlier_info
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)."""
        score = 100.0
        
        # Deduct points for missing data
        missing_data = results["missing_data"]
        if missing_data["total_missing_values"] > 0:
            missing_penalty = min(30, len(missing_data["columns_with_missing"]) * 5)
            score -= missing_penalty
        
        # Deduct points for duplicates
        duplicate_penalty = min(20, results["duplicates"]["duplicate_percentage"])
        score -= duplicate_penalty
        
        # Deduct points for potential data type issues
        type_issues = len(results["data_types"]["potential_issues"])
        score -= min(15, type_issues * 3)
        
        return max(0.0, round(score, 1))
    
    def save_validation_report(self, output_path: str):
        """Save validation results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        self.logger.info(f"Validation report saved to {output_path}")
    
    def print_summary(self, dataset_name: str):
        """Print a summary of validation results."""
        if dataset_name not in self.validation_results:
            self.logger.error(f"No validation results found for {dataset_name}")
            return
        
        results = self.validation_results[dataset_name]
        
        print(f"\n📊 Data Validation Summary for {dataset_name}")
        print("=" * 50)
        print(f"Quality Score: {results['data_quality_score']}/100")
        print(f"Total Rows: {results['basic_info']['total_rows']:,}")
        print(f"Total Columns: {results['basic_info']['total_columns']}")
        print(f"Memory Usage: {results['basic_info']['memory_usage_mb']} MB")
        
        print(f"\n🔍 Missing Data:")
        missing = results['missing_data']
        print(f"  Total Missing Values: {missing['total_missing_values']:,}")
        print(f"  Columns with Missing: {len(missing['columns_with_missing'])}")
        
        print(f"\n📋 Duplicates:")
        duplicates = results['duplicates']
        print(f"  Duplicate Rows: {duplicates['total_duplicate_rows']:,}")
        print(f"  Duplicate Percentage: {duplicates['duplicate_percentage']}%")
        
        print(f"\n⚠️  Data Type Issues: {len(results['data_types']['potential_issues'])}")
        
        if results['data_quality_score'] >= 80:
            print("✅ Data quality is GOOD")
        elif results['data_quality_score'] >= 60:
            print("⚠️  Data quality is FAIR - some issues need attention")
        else:
            print("❌ Data quality is POOR - significant issues detected")