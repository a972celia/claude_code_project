"""
Model training infrastructure for the AI-Powered Underwriting Engine.
Implements multiple ML algorithms with evaluation, tuning, and explainability.
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

# Explainability
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class UnderwritingModelTrainer:
    """
    Comprehensive model training and evaluation for loan underwriting.
    """
    
    def __init__(self, 
                 models_dir: str = "models",
                 results_dir: str = "results",
                 random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            models_dir: Directory to save trained models
            results_dir: Directory to save results and reports
            random_state: Random seed for reproducibility
        """
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.random_state = random_state
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for models and results
        self.models = {}
        self.results = {}
        self.feature_names = None
        
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Load preprocessed training data.
        
        Args:
            data_dir: Directory containing processed data
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        data_path = Path(data_dir)
        
        self.logger.info(f"Loading processed data from {data_path}")
        
        # Load datasets
        X_train = pd.read_parquet(data_path / "X_train.parquet")
        X_val = pd.read_parquet(data_path / "X_val.parquet") 
        X_test = pd.read_parquet(data_path / "X_test.parquet")
        y_train = pd.read_parquet(data_path / "y_train.parquet").iloc[:, 0]
        y_val = pd.read_parquet(data_path / "y_val.parquet").iloc[:, 0]
        y_test = pd.read_parquet(data_path / "y_test.parquet").iloc[:, 0]
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        self.logger.info(f"Data loaded successfully:")
        self.logger.info(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        self.logger.info(f"  Val:   {X_val.shape[0]:,} samples")
        self.logger.info(f"  Test:  {X_test.shape[0]:,} samples")
        self.logger.info(f"  Default rates: Train {y_train.mean():.3f}, Val {y_val.mean():.3f}, Test {y_test.mean():.3f}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_baseline_models(self) -> Dict[str, Any]:
        """
        Get baseline model configurations.
        
        Returns:
            Dictionary of model name to model instance
        """
        models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            )
        }
        
        return models
    
    def evaluate_model(self, 
                      model: Any, 
                      X_train: pd.DataFrame, 
                      X_val: pd.DataFrame, 
                      X_test: pd.DataFrame,
                      y_train: pd.Series, 
                      y_val: pd.Series, 
                      y_test: pd.Series,
                      model_name: str) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_train, X_val, X_test: Feature sets
            y_train, y_val, y_test: Target sets
            model_name: Name of the model
            
        Returns:
            Dictionary containing evaluation results
        """
        results = {
            'model_name': model_name,
            'train_metrics': {},
            'val_metrics': {},
            'test_metrics': {},
            'feature_importance': None,
            'training_time': None
        }
        
        # Predictions for all sets
        datasets = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        for set_name, (X, y) in datasets.items():
            # Predictions
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1_score': f1_score(y, y_pred),
                'roc_auc': roc_auc_score(y, y_pred_proba),
                'samples': len(y),
                'default_rate': y.mean()
            }
            
            results[f'{set_name}_metrics'] = metrics
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = importance_df.to_dict('records')
        elif hasattr(model, 'coef_'):
            # For logistic regression
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(model.coef_[0])
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = importance_df.to_dict('records')
        
        self.logger.info(f"Model {model_name} evaluation completed")
        self.logger.info(f"  Val AUC: {results['val_metrics']['roc_auc']:.4f}")
        self.logger.info(f"  Test AUC: {results['test_metrics']['roc_auc']:.4f}")
        
        return results
    
    def cross_validate_model(self, 
                           model: Any, 
                           X: pd.DataFrame, 
                           y: pd.Series,
                           cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            model: Model to validate
            X: Features
            y: Target
            cv_folds: Number of CV folds
            
        Returns:
            CV results
        """
        cv_scores = cross_val_score(
            model, X, y, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def train_baseline_models(self, 
                            X_train: pd.DataFrame, 
                            X_val: pd.DataFrame, 
                            X_test: pd.DataFrame,
                            y_train: pd.Series, 
                            y_val: pd.Series, 
                            y_test: pd.Series) -> Dict[str, Any]:
        """
        Train all baseline models.
        
        Args:
            X_train, X_val, X_test: Feature sets
            y_train, y_val, y_test: Target sets
            
        Returns:
            Training results for all models
        """
        models = self.get_baseline_models()
        training_results = {}
        
        self.logger.info(f"Training {len(models)} baseline models...")
        
        for model_name, model in models.items():
            self.logger.info(f"Training {model_name}...")
            start_time = time.time()
            
            # Train model
            if model_name == 'xgboost':
                # XGBoost early stopping
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            elif model_name == 'lightgbm':
                # LightGBM early stopping
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
            else:
                model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Evaluate model
            results = self.evaluate_model(
                model, X_train, X_val, X_test, 
                y_train, y_val, y_test, model_name
            )
            results['training_time'] = training_time
            
            # Cross-validation
            cv_results = self.cross_validate_model(model, X_train, y_train)
            results['cross_validation'] = cv_results
            
            # Store model and results
            self.models[model_name] = model
            self.results[model_name] = results
            training_results[model_name] = results
            
            self.logger.info(f"  {model_name} completed in {training_time:.2f}s")
            self.logger.info(f"  CV AUC: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
        
        return training_results
    
    def hyperparameter_tuning(self, 
                            model_name: str,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_val: pd.DataFrame,
                            y_val: pd.Series,
                            param_grid: Dict[str, List]) -> Tuple[Any, Dict[str, Any]]:
        """
        Simple grid search for hyperparameter tuning.
        
        Args:
            model_name: Name of model to tune
            X_train, y_train: Training data
            X_val, y_val: Validation data
            param_grid: Parameter grid to search
            
        Returns:
            Best model and tuning results
        """
        self.logger.info(f"Hyperparameter tuning for {model_name}...")
        
        best_score = 0
        best_params = None
        best_model = None
        tuning_results = []
        
        # Get base model
        models = self.get_baseline_models()
        base_model = models[model_name]
        
        # Generate parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))
            
            # Set parameters
            for param, value in params.items():
                setattr(base_model, param, value)
            
            # Train and evaluate
            if model_name == 'xgboost':
                base_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            elif model_name == 'lightgbm':
                base_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
            else:
                base_model.fit(X_train, y_train)
            
            # Score on validation set
            val_pred_proba = base_model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, val_pred_proba)
            
            tuning_results.append({
                'params': params.copy(),
                'val_auc': score
            })
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_model = pickle.loads(pickle.dumps(base_model))  # Deep copy
        
        self.logger.info(f"Best parameters for {model_name}: {best_params}")
        self.logger.info(f"Best validation AUC: {best_score:.4f}")
        
        return best_model, {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': tuning_results
        }
    
    def generate_shap_explanations(self, 
                                 model: Any, 
                                 X_sample: pd.DataFrame,
                                 model_name: str,
                                 max_samples: int = 1000) -> Dict[str, Any]:
        """
        Generate SHAP explanations for model interpretability.
        
        Args:
            model: Trained model
            X_sample: Sample data for explanations
            model_name: Name of the model
            max_samples: Maximum samples for SHAP calculation
            
        Returns:
            SHAP analysis results
        """
        self.logger.info(f"Generating SHAP explanations for {model_name}...")
        
        # Sample data if too large
        if len(X_sample) > max_samples:
            X_sample = X_sample.sample(n=max_samples, random_state=self.random_state)
        
        try:
            # Create SHAP explainer
            if model_name in ['xgboost', 'lightgbm', 'random_forest']:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.LinearExplainer(model, X_sample)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # For binary classification, take the positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Feature importance (mean absolute SHAP values)
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.mean(np.abs(shap_values), axis=0)
            }).sort_values('importance', ascending=False)
            
            results = {
                'feature_importance': feature_importance.to_dict('records'),
                'shap_values_shape': shap_values.shape,
                'explanation_samples': len(X_sample)
            }
            
            self.logger.info(f"SHAP explanations generated for {len(X_sample)} samples")
            
            return results
            
        except Exception as e:
            self.logger.warning(f"SHAP explanation failed for {model_name}: {str(e)}")
            return {'error': str(e)}
    
    def save_models_and_results(self):
        """Save trained models and results."""
        self.logger.info("Saving models and results...")
        
        # Save models
        for model_name, model in self.models.items():
            model_path = self.models_dir / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            self.logger.info(f"Saved model: {model_path}")
        
        # Save results
        results_path = self.results_dir / "training_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {}
            for model_name, results in self.results.items():
                json_results[model_name] = self._convert_to_json_serializable(results)
            json.dump(json_results, f, indent=2)
        self.logger.info(f"Saved results: {results_path}")
        
        # Save model comparison
        self.create_model_comparison_report()
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def create_model_comparison_report(self):
        """Create a comprehensive model comparison report."""
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'model': model_name,
                'train_auc': results['train_metrics']['roc_auc'],
                'val_auc': results['val_metrics']['roc_auc'],
                'test_auc': results['test_metrics']['roc_auc'],
                'val_precision': results['val_metrics']['precision'],
                'val_recall': results['val_metrics']['recall'],
                'val_f1': results['val_metrics']['f1_score'],
                'cv_auc_mean': results['cross_validation']['cv_mean'],
                'cv_auc_std': results['cross_validation']['cv_std'],
                'training_time': results['training_time']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('val_auc', ascending=False)
        
        # Save comparison table
        comparison_path = self.results_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        self.logger.info(f"Saved model comparison: {comparison_path}")
        
        return comparison_df
    
    def get_best_model(self) -> Tuple[str, Any, Dict[str, Any]]:
        """
        Get the best performing model based on validation AUC.
        
        Returns:
            Model name, model instance, and results
        """
        best_model_name = None
        best_val_auc = 0
        
        for model_name, results in self.results.items():
            val_auc = results['val_metrics']['roc_auc']
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_name = model_name
        
        if best_model_name:
            return (best_model_name, 
                    self.models[best_model_name], 
                    self.results[best_model_name])
        else:
            raise ValueError("No models have been trained yet")
    
    def generate_training_summary(self) -> str:
        """Generate a comprehensive training summary."""
        summary = []
        summary.append("🤖 MODEL TRAINING SUMMARY")
        summary.append("=" * 50)
        
        if not self.results:
            summary.append("No models have been trained yet.")
            return "\n".join(summary)
        
        # Best model
        best_name, _, best_results = self.get_best_model()
        summary.append(f"🏆 Best Model: {best_name}")
        summary.append(f"   Validation AUC: {best_results['val_metrics']['roc_auc']:.4f}")
        summary.append(f"   Test AUC: {best_results['test_metrics']['roc_auc']:.4f}")
        
        # All models comparison
        summary.append(f"\n📊 All Models Performance:")
        for model_name, results in sorted(self.results.items(), 
                                        key=lambda x: x[1]['val_metrics']['roc_auc'], 
                                        reverse=True):
            val_auc = results['val_metrics']['roc_auc']
            test_auc = results['test_metrics']['roc_auc']
            cv_auc = results['cross_validation']['cv_mean']
            cv_std = results['cross_validation']['cv_std']
            training_time = results['training_time']
            
            summary.append(f"   {model_name:15} | Val: {val_auc:.4f} | Test: {test_auc:.4f} | "
                          f"CV: {cv_auc:.4f}±{cv_std:.4f} | Time: {training_time:.1f}s")
        
        # Feature importance (from best model)
        if 'feature_importance' in best_results and best_results['feature_importance']:
            summary.append(f"\n🎯 Top 10 Features ({best_name}):")
            for i, feat in enumerate(best_results['feature_importance'][:10]):
                summary.append(f"   {i+1:2d}. {feat['feature']:25} | {feat['importance']:.4f}")
        
        return "\n".join(summary)