"""
Exploratory Data Analysis utilities for the AI-Powered Underwriting Engine.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class EDAAnalyzer:
    """Class for comprehensive exploratory data analysis of loan datasets."""
    
    def __init__(self, df: pd.DataFrame, figsize_default: Tuple[int, int] = (12, 8)):
        """
        Initialize EDA analyzer.
        
        Args:
            df: DataFrame to analyze
            figsize_default: Default figure size for plots
        """
        self.df = df.copy()
        self.figsize_default = figsize_default
        self.target_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.date_columns = []
        
        # Identify column types
        self._identify_column_types()
        
    def _identify_column_types(self):
        """Identify different types of columns in the dataset."""
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                self.numeric_columns.append(col)
            elif self.df[col].dtype == 'object':
                # Check if it's a date column
                if any(date_indicator in col.lower() for date_indicator in ['date', 'time', 'on']):
                    self.date_columns.append(col)
                else:
                    self.categorical_columns.append(col)
            elif self.df[col].dtype == 'bool':
                self.categorical_columns.append(col)
        
        # Identify potential target columns
        potential_targets = ['default', 'status', 'outcome', 'target', 'label']
        for col in self.df.columns:
            if any(target in col.lower() for target in potential_targets):
                self.target_columns.append(col)
    
    def basic_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        info = {
            'shape': self.df.shape,
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'duplicate_rows': self.df.duplicated().sum(),
            'column_types': {
                'numeric': len(self.numeric_columns),
                'categorical': len(self.categorical_columns),
                'date': len(self.date_columns)
            },
            'missing_data': {
                'total_missing': self.df.isnull().sum().sum(),
                'columns_with_missing': self.df.isnull().sum()[self.df.isnull().sum() > 0].shape[0],
                'missing_percentage': round((self.df.isnull().sum().sum() / self.df.size) * 100, 2)
            }
        }
        return info
    
    def missing_data_analysis(self) -> pd.DataFrame:
        """Analyze missing data patterns."""
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df)) * 100,
            'Data_Type': self.df.dtypes
        })
        
        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        )
        
        return missing_data
    
    def plot_missing_data(self, top_n: int = 20):
        """Visualize missing data patterns."""
        missing_data = self.missing_data_analysis()
        
        if len(missing_data) == 0:
            print("No missing data found!")
            return
            
        # Plot top N columns with missing data
        top_missing = missing_data.head(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of missing percentages
        axes[0].barh(range(len(top_missing)), top_missing['Missing_Percentage'])
        axes[0].set_yticks(range(len(top_missing)))
        axes[0].set_yticklabels(top_missing['Column'])
        axes[0].set_xlabel('Missing Percentage (%)')
        axes[0].set_title(f'Top {len(top_missing)} Columns with Missing Data')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Heatmap of missing data pattern
        missing_matrix = self.df[top_missing['Column'].head(15)].isnull()
        sns.heatmap(missing_matrix.T, cbar=True, cmap='viridis', 
                   yticklabels=True, xticklabels=False, ax=axes[1])
        axes[1].set_title('Missing Data Pattern (Sample)')
        
        plt.tight_layout()
        plt.show()
        
        return missing_data
    
    def analyze_target_variable(self, target_col: str):
        """Analyze the target variable distribution."""
        if target_col not in self.df.columns:
            print(f"Column '{target_col}' not found in dataset!")
            return None
            
        print(f"📊 Target Variable Analysis: {target_col}")
        print("=" * 50)
        
        # Basic statistics
        target_stats = {
            'unique_values': self.df[target_col].nunique(),
            'null_count': self.df[target_col].isnull().sum(),
            'value_counts': self.df[target_col].value_counts().head(10)
        }
        
        print(f"Unique values: {target_stats['unique_values']}")
        print(f"Missing values: {target_stats['null_count']}")
        print("\nValue counts:")
        print(target_stats['value_counts'])
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Value counts bar plot
        value_counts = self.df[target_col].value_counts().head(10)
        axes[0].bar(range(len(value_counts)), value_counts.values)
        axes[0].set_xticks(range(len(value_counts)))
        axes[0].set_xticklabels(value_counts.index, rotation=45)
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'{target_col} Distribution')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Pie chart for proportions
        if len(value_counts) <= 10:
            axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            axes[1].set_title(f'{target_col} Proportions')
        else:
            # If too many categories, show top 5 and "Others"
            top_5 = value_counts.head(5)
            others = value_counts.iloc[5:].sum()
            if others > 0:
                pie_data = list(top_5.values) + [others]
                pie_labels = list(top_5.index) + ['Others']
            else:
                pie_data = top_5.values
                pie_labels = top_5.index
            axes[1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%')
            axes[1].set_title(f'{target_col} Proportions (Top 5)')
        
        plt.tight_layout()
        plt.show()
        
        return target_stats
    
    def numeric_summary(self) -> pd.DataFrame:
        """Get summary statistics for numeric columns."""
        if not self.numeric_columns:
            print("No numeric columns found!")
            return pd.DataFrame()
            
        summary = self.df[self.numeric_columns].describe()
        
        # Add additional statistics
        additional_stats = pd.DataFrame({
            'missing_count': self.df[self.numeric_columns].isnull().sum(),
            'missing_pct': (self.df[self.numeric_columns].isnull().sum() / len(self.df)) * 100,
            'skewness': self.df[self.numeric_columns].skew(),
            'kurtosis': self.df[self.numeric_columns].kurtosis()
        }).T
        
        summary = pd.concat([summary, additional_stats])
        return summary.round(3)
    
    def plot_numeric_distributions(self, columns: List[str] = None, n_cols: int = 3):
        """Plot distributions of numeric variables."""
        if columns is None:
            columns = self.numeric_columns[:12]  # Limit to first 12
        
        n_plots = len(columns)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
            
        for i, col in enumerate(columns):
            row, col_idx = i // n_cols, i % n_cols
            
            # Skip missing values for plotting
            data = self.df[col].dropna()
            
            if len(data) > 0:
                axes[row, col_idx].hist(data, bins=50, alpha=0.7, edgecolor='black')
                axes[row, col_idx].set_title(f'{col}\n(n={len(data)})')
                axes[row, col_idx].set_ylabel('Frequency')
                axes[row, col_idx].grid(alpha=0.3)
            else:
                axes[row, col_idx].text(0.5, 0.5, 'No Data', ha='center', va='center')
                axes[row, col_idx].set_title(f'{col}\n(No Data)')
        
        # Hide empty subplots
        for i in range(n_plots, n_rows * n_cols):
            row, col_idx = i // n_cols, i % n_cols
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self, target_col: str = None, threshold: float = 0.5):
        """Analyze correlations between numeric variables."""
        if not self.numeric_columns:
            print("No numeric columns for correlation analysis!")
            return None
            
        # Calculate correlation matrix
        corr_matrix = self.df[self.numeric_columns].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numeric Variables')
        plt.tight_layout()
        plt.show()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False)
        
        if len(high_corr_df) > 0:
            print(f"\n🔍 High Correlations (|r| > {threshold}):")
            print(high_corr_df.to_string(index=False))
        else:
            print(f"\n✅ No high correlations found (|r| > {threshold})")
        
        # If target column specified, show correlations with target
        if target_col and target_col in self.numeric_columns:
            target_corrs = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
            print(f"\n🎯 Correlations with {target_col}:")
            print(target_corrs.head(10).to_string())
        
        return corr_matrix, high_corr_df
    
    def categorical_summary(self) -> Dict[str, Any]:
        """Summarize categorical variables."""
        if not self.categorical_columns:
            print("No categorical columns found!")
            return {}
            
        cat_summary = {}
        for col in self.categorical_columns:
            cat_summary[col] = {
                'unique_count': self.df[col].nunique(),
                'missing_count': self.df[col].isnull().sum(),
                'top_values': self.df[col].value_counts().head(5).to_dict()
            }
        
        return cat_summary
    
    def generate_eda_report(self) -> str:
        """Generate a comprehensive EDA report."""
        report = []
        report.append("📊 EXPLORATORY DATA ANALYSIS REPORT")
        report.append("=" * 50)
        
        # Basic info
        basic_info = self.basic_info()
        report.append(f"\n📋 Dataset Overview:")
        report.append(f"   Shape: {basic_info['shape']} (rows × columns)")
        report.append(f"   Memory usage: {basic_info['memory_usage_mb']} MB")
        report.append(f"   Duplicate rows: {basic_info['duplicate_rows']}")
        
        # Column types
        report.append(f"\n📊 Column Types:")
        report.append(f"   Numeric: {basic_info['column_types']['numeric']}")
        report.append(f"   Categorical: {basic_info['column_types']['categorical']}")
        report.append(f"   Date: {basic_info['column_types']['date']}")
        
        # Missing data
        report.append(f"\n🔍 Missing Data:")
        report.append(f"   Total missing values: {basic_info['missing_data']['total_missing']:,}")
        report.append(f"   Columns with missing data: {basic_info['missing_data']['columns_with_missing']}")
        report.append(f"   Overall missing percentage: {basic_info['missing_data']['missing_percentage']}%")
        
        # Potential targets
        if self.target_columns:
            report.append(f"\n🎯 Potential Target Columns:")
            for col in self.target_columns:
                report.append(f"   - {col}")
        
        return "\n".join(report)