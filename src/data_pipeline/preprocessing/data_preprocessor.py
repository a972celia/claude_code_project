"""
Data preprocessing pipeline for the AI-Powered Underwriting Engine.
Implements comprehensive data cleaning, feature engineering, and preparation based on EDA insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

warnings.filterwarnings('ignore')

class LoanDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for loan underwriting data.
    Based on EDA insights from Bondora P2P loans dataset.
    """
    
    def __init__(self, 
                 missing_threshold: float = 0.6,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42):
        """
        Initialize the preprocessor.
        
        Args:
            missing_threshold: Drop features with missing data > this threshold
            test_size: Proportion for test set
            val_size: Proportion for validation set  
            random_state: Random seed for reproducibility
        """
        self.missing_threshold = missing_threshold
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Feature lists based on EDA insights
        self.features_to_drop = []
        self.numeric_features = []
        self.categorical_features = []
        self.engineered_features = []
        
        # Preprocessing artifacts
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        
        self.logger = logging.getLogger(__name__)
        
    def identify_features_to_drop(self, df: pd.DataFrame) -> List[str]:
        """
        Identify features to drop based on missing data threshold and EDA insights.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of feature names to drop
        """
        # Calculate missing percentages
        missing_pct = (df.isnull().sum() / len(df))
        high_missing = missing_pct[missing_pct > self.missing_threshold].index.tolist()
        
        # Additional features to drop based on EDA insights
        additional_drops = [
            'LoanId',  # Identifier, not predictive
            'LoanNumber',  # Identifier
            # Features with >60% missing identified in EDA
            'CreditScoreEsEquifaxRisk',
            'LoanCancelled', 
            'PlannedPrincipalTillDate',
            'PreviousEarlyRepaymentsBeforeLoan',
            'GracePeriodEnd',
            'GracePeriodStart',
            'ContractEndDate',
            'InterestAndPenaltyWriteOffs',
            'PrincipalDebtServicingCost',
            'PrincipalWriteOffs',
            'InterestAndPenaltyDebtServicingCost',
            'DefaultDate',  # Will be used to create target
            'PrincipalRecovery',
            'PlannedPrincipalPostDefault',
            'InterestRecovery',
            'PlannedInterestPostDefault',
            'EAD1',
            'EAD2',
            'ReScheduledOn',
            'ActiveLateCategory',
        ]
        
        # Combine and deduplicate
        all_drops = list(set(high_missing + additional_drops))
        
        # Only drop features that actually exist in the dataset
        features_to_drop = [col for col in all_drops if col in df.columns]
        
        self.logger.info(f"Identified {len(features_to_drop)} features to drop")
        return features_to_drop
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """
        Create target variable based on EDA insights.
        Target: Late payments or loans with default dates.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Binary target variable (1 = default/late, 0 = good)
        """
        # Create target based on EDA findings
        target = pd.Series(0, index=df.index, name='target')
        
        # Mark as default if:
        # 1. Status contains "Late"
        # 2. Has a DefaultDate 
        late_status = df['Status'].str.contains('Late', na=False)
        has_default_date = df['DefaultDate'].notna()
        
        target = (late_status | has_default_date).astype(int)
        
        default_rate = target.mean()
        self.logger.info(f"Target variable created: {target.sum():,} defaults "
                        f"({default_rate:.2%} default rate)")
        
        return target
    
    def handle_mixed_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle columns with mixed data types identified in EDA.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with fixed data types
        """
        df = df.copy()
        
        # Mixed type columns identified in EDA (columns 10,56,73,74,94,95)
        mixed_type_cols = [
            'EmploymentDurationCurrentEmployer',  # Column ~10
            'CreditScoreEsMicroL',  # Column ~56  
            'CreditScoreFiAsiakasTietoRiskGrade',  # Column ~73
            'CreditScoreEsEquifaxRisk',  # Column ~74
            'ActiveLateLastPaymentCategory',  # Column ~94
            'PreviousEarlyRepaymentsBeforeLoan'  # Column ~95
        ]
        
        for col in mixed_type_cols:
            if col in df.columns:
                # Convert to string first, then handle appropriately
                df[col] = df[col].astype(str)
                
                # Try to convert to numeric if possible
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers identified in EDA.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()
        
        # Handle Age = 0 (invalid ages)
        if 'Age' in df.columns:
            # Replace Age = 0 with median age
            median_age = df[df['Age'] > 0]['Age'].median()
            df.loc[df['Age'] == 0, 'Age'] = median_age
            self.logger.info(f"Replaced {(df['Age'] == 0).sum()} zero ages with median: {median_age}")
        
        # Handle extreme income values (likely data entry errors)
        income_cols = ['IncomeTotal', 'IncomeFromPrincipalEmployer']
        for col in income_cols:
            if col in df.columns:
                # Cap at 99th percentile
                cap_value = df[col].quantile(0.99)
                outliers = df[col] > cap_value
                df.loc[outliers, col] = cap_value
                self.logger.info(f"Capped {outliers.sum()} outliers in {col} at {cap_value:.2f}")
        
        # Handle extreme debt-to-income ratios
        if 'DebtToIncome' in df.columns:
            # Cap at 99th percentile
            cap_value = df['DebtToIncome'].quantile(0.99)
            outliers = df['DebtToIncome'] > cap_value
            df.loc[outliers, 'DebtToIncome'] = cap_value
            self.logger.info(f"Capped {outliers.sum()} extreme debt-to-income ratios at {cap_value:.2f}")
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features based on EDA insights.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # 1. Financial ratios and risk indicators
        if 'MonthlyPayment' in df.columns and 'IncomeTotal' in df.columns:
            df['payment_to_income_ratio'] = df['MonthlyPayment'] / (df['IncomeTotal'] + 1)
            self.engineered_features.append('payment_to_income_ratio')
        
        if 'Amount' in df.columns and 'IncomeTotal' in df.columns:
            df['loan_to_income_ratio'] = df['Amount'] / (df['IncomeTotal'] + 1)
            self.engineered_features.append('loan_to_income_ratio')
        
        # 2. Interest rate categories (based on EDA insights)
        if 'Interest' in df.columns:
            df['interest_category'] = pd.cut(df['Interest'], 
                                           bins=[0, 10, 20, 30, 40, 100],
                                           labels=['low', 'medium', 'high', 'very_high', 'extreme'])
            self.engineered_features.append('interest_category')
        
        # 3. Age categories
        if 'Age' in df.columns:
            df['age_category'] = pd.cut(df['Age'],
                                      bins=[0, 25, 35, 45, 55, 100],
                                      labels=['young', 'adult', 'middle', 'senior', 'elderly'])
            self.engineered_features.append('age_category')
        
        # 4. Loan amount categories
        if 'Amount' in df.columns:
            df['amount_category'] = pd.cut(df['Amount'],
                                         bins=[0, 1000, 2000, 3000, 5000, 10000, float('inf')],
                                         labels=['micro', 'small', 'medium', 'large', 'xlarge', 'jumbo'])
            self.engineered_features.append('amount_category')
        
        # 5. Income diversification
        income_sources = ['IncomeFromPrincipalEmployer', 'IncomeFromPension', 
                         'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare',
                         'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther']
        
        available_income_sources = [col for col in income_sources if col in df.columns]
        if len(available_income_sources) > 1:
            df['income_sources_count'] = (df[available_income_sources] > 0).sum(axis=1)
            self.engineered_features.append('income_sources_count')
        
        # 6. Credit score availability indicator
        credit_score_cols = [col for col in df.columns if 'CreditScore' in col]
        if credit_score_cols:
            df['has_credit_score'] = df[credit_score_cols].notna().any(axis=1).astype(int)
            self.engineered_features.append('has_credit_score')
        
        # 7. Previous loan history indicator
        prev_loan_cols = ['NoOfPreviousLoansBeforeLoan', 'AmountOfPreviousLoansBeforeLoan']
        available_prev_cols = [col for col in prev_loan_cols if col in df.columns]
        if available_prev_cols:
            df['has_previous_loans'] = (df[available_prev_cols] > 0).any(axis=1).astype(int)
            self.engineered_features.append('has_previous_loans')
        
        # 8. Time-based features from date columns
        date_cols = ['LoanApplicationStartedDate', 'LoanDate']
        for col in date_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_weekday'] = df[col].dt.dayofweek
                    self.engineered_features.extend([f'{col}_year', f'{col}_month', f'{col}_weekday'])
                except:
                    continue
        
        self.logger.info(f"Created {len(self.engineered_features)} engineered features")
        return df
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select final features for modeling based on EDA insights.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of selected feature names
        """
        # Core features with low missing data (from EDA)
        core_features = [
            'Amount', 'Interest', 'LoanDuration', 'MonthlyPayment',
            'Age', 'IncomeTotal', 'DebtToIncome',
            'ApplicationSignedHour', 'ApplicationSignedWeekday',
            'VerificationType', 'LanguageCode', 'Education',
            'HomeOwnershipType', 'Country', 'Rating'
        ]
        
        # Additional features with acceptable missing data
        secondary_features = [
            'Gender', 'AppliedAmount', 'ExistingLiabilities',
            'LiabilitiesTotal', 'RefinanceLiabilities', 'FreeCash',
            'MonthlyPaymentDay', 'CreditScoreEsMicroL',
            'NoOfPreviousLoansBeforeLoan', 'AmountOfPreviousLoansBeforeLoan'
        ]
        
        # Combine with engineered features
        all_candidate_features = core_features + secondary_features + self.engineered_features
        
        # Only include features that exist in the dataset
        selected_features = [col for col in all_candidate_features if col in df.columns]
        
        # Separate numeric and categorical features
        self.numeric_features = []
        self.categorical_features = []
        
        for col in selected_features:
            if df[col].dtype in ['int64', 'float64']:
                self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)
        
        self.logger.info(f"Selected {len(selected_features)} features: "
                        f"{len(self.numeric_features)} numeric, {len(self.categorical_features)} categorical")
        
        return selected_features
    
    def handle_missing_values(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Handle missing values in selected features.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data (for fitting strategies)
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # Handle numeric features
        for col in self.numeric_features:
            if col in df.columns:
                if is_training:
                    # Store median for future use
                    self.feature_stats[f'{col}_median'] = df[col].median()
                
                # Fill with stored median
                fill_value = self.feature_stats.get(f'{col}_median', df[col].median())
                df[col] = df[col].fillna(fill_value)
        
        # Handle categorical features
        for col in self.categorical_features:
            if col in df.columns:
                if is_training:
                    # Store mode for future use
                    mode_value = df[col].mode()
                    self.feature_stats[f'{col}_mode'] = mode_value[0] if len(mode_value) > 0 else 'unknown'
                
                # Fill with stored mode
                fill_value = self.feature_stats.get(f'{col}_mode', 'unknown')
                df[col] = df[col].fillna(fill_value)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data (for fitting encoders)
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        
        for col in self.categorical_features:
            if col in df.columns:
                if is_training:
                    # Fit encoder
                    encoder = LabelEncoder()
                    encoder.fit(df[col].astype(str))
                    self.encoders[col] = encoder
                
                # Transform
                if col in self.encoders:
                    try:
                        df[col] = self.encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        df[col] = df[col].astype(str)
                        known_labels = set(self.encoders[col].classes_)
                        df[col] = df[col].apply(lambda x: x if x in known_labels else 'unknown')
                        df[col] = self.encoders[col].transform(df[col])
        
        return df
    
    def scale_numeric_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Scale numeric features.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data (for fitting scalers)
            
        Returns:
            DataFrame with scaled numeric features
        """
        df = df.copy()
        
        if is_training:
            # Fit scaler
            scaler = StandardScaler()
            scaler.fit(df[self.numeric_features])
            self.scalers['standard'] = scaler
        
        # Transform
        if 'standard' in self.scalers:
            df[self.numeric_features] = self.scalers['standard'].transform(df[self.numeric_features])
        
        return df
    
    def create_splits(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Create train/validation/test splits.
        
        Args:
            X: Features
            y: Target variable
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state, stratify=y_temp
        )
        
        self.logger.info(f"Data splits created:")
        self.logger.info(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X):.1%})")
        self.logger.info(f"  Val:   {len(X_val):,} samples ({len(X_val)/len(X):.1%})")
        self.logger.info(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(X):.1%})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Complete preprocessing pipeline for training data.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        self.logger.info("Starting preprocessing pipeline...")
        
        # 1. Handle mixed data types
        df = self.handle_mixed_data_types(df)
        
        # 2. Create target variable
        y = self.create_target_variable(df)
        
        # 3. Identify and drop unusable features
        self.features_to_drop = self.identify_features_to_drop(df)
        df = df.drop(columns=self.features_to_drop)
        
        # 4. Handle outliers
        df = self.handle_outliers(df)
        
        # 5. Feature engineering
        df = self.feature_engineering(df)
        
        # 6. Select final features
        selected_features = self.select_features(df)
        X = df[selected_features].copy()
        
        # 7. Create train/val/test splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_splits(X, y)
        
        # 8. Handle missing values (fit on training data)
        X_train = self.handle_missing_values(X_train, is_training=True)
        X_val = self.handle_missing_values(X_val, is_training=False)
        X_test = self.handle_missing_values(X_test, is_training=False)
        
        # 9. Encode categorical features (fit on training data)
        X_train = self.encode_categorical_features(X_train, is_training=True)
        X_val = self.encode_categorical_features(X_val, is_training=False)
        X_test = self.encode_categorical_features(X_test, is_training=False)
        
        # 10. Scale numeric features (fit on training data)
        X_train = self.scale_numeric_features(X_train, is_training=True)
        X_val = self.scale_numeric_features(X_val, is_training=False)
        X_test = self.scale_numeric_features(X_test, is_training=False)
        
        self.logger.info("Preprocessing pipeline completed successfully!")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessing pipeline.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Processed DataFrame
        """
        # Apply same transformations as training (except target creation and splitting)
        df = self.handle_mixed_data_types(df)
        df = df.drop(columns=self.features_to_drop, errors='ignore')
        df = self.handle_outliers(df)
        df = self.feature_engineering(df)
        
        # Select same features as training
        available_features = [col for col in (self.numeric_features + self.categorical_features) if col in df.columns]
        X = df[available_features].copy()
        
        # Apply transformations
        X = self.handle_missing_values(X, is_training=False)
        X = self.encode_categorical_features(X, is_training=False)
        X = self.scale_numeric_features(X, is_training=False)
        
        return X
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing pipeline.
        
        Returns:
            Dictionary with pipeline summary
        """
        return {
            'features_dropped': len(self.features_to_drop),
            'features_selected': len(self.numeric_features) + len(self.categorical_features),
            'numeric_features': len(self.numeric_features),
            'categorical_features': len(self.categorical_features),
            'engineered_features': len(self.engineered_features),
            'missing_threshold': self.missing_threshold,
            'preprocessing_steps': [
                'Mixed data types handling',
                'Target variable creation',
                'Feature dropping',
                'Outlier handling',
                'Feature engineering',
                'Feature selection',
                'Missing value imputation',
                'Categorical encoding',
                'Numeric scaling'
            ]
        }