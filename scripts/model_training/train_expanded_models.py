#!/usr/bin/env python3
"""
Train Models with Expanded Free Review Features

This script trains ML models using the expanded feature set (66 features vs 38 baseline)
and compares performance against the baseline models.

Expected Improvements:
- XGBoost: 0.808 → 0.83-0.85 AUC (+2-5%)
- Enhanced explainability with sentiment features
- Better business risk assessment
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import json
from datetime import datetime
import time

# ML imports
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def setup_logging():
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('expanded_model_training.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

def load_expanded_data():
    """Load the expanded dataset with enhanced features."""
    logger = logging.getLogger(__name__)
    
    data_files = {
        'X_train': 'data/processed_expanded/expanded_X_train_enhanced.parquet',
        'X_val': 'data/processed_expanded/expanded_X_val_enhanced.parquet', 
        'X_test': 'data/processed_expanded/expanded_X_test_enhanced.parquet',
        'y_train': 'data/processed_enhanced/y_train_enhanced.parquet',
        'y_val': 'data/processed_enhanced/y_val_enhanced.parquet',
        'y_test': 'data/processed_enhanced/y_test_enhanced.parquet'
    }
    
    datasets = {}
    
    for name, file_path in data_files.items():
        path = Path(file_path)
        if not path.exists():
            logger.error(f"❌ Data file not found: {file_path}")
            return None
        
        try:
            datasets[name] = pd.read_parquet(path)
            logger.info(f"✅ Loaded {name}: {datasets[name].shape}")
        except Exception as e:
            logger.error(f"❌ Failed to load {name}: {str(e)}")
            return None
    
    # Verify shapes match
    if len(datasets['X_train']) != len(datasets['y_train']):
        logger.error("❌ Training data shape mismatch")
        return None
    
    logger.info(f"📊 Dataset Summary:")
    logger.info(f"   Training: {len(datasets['X_train']):,} samples, {len(datasets['X_train'].columns)} features")
    logger.info(f"   Validation: {len(datasets['X_val']):,} samples")
    logger.info(f"   Test: {len(datasets['X_test']):,} samples")
    
    return datasets

def analyze_feature_importance(datasets):
    """Analyze the new features added."""
    logger = logging.getLogger(__name__)
    
    X_train = datasets['X_train']
    
    # Identify new review features
    review_features = [col for col in X_train.columns if col.startswith('review_')]
    baseline_features = [col for col in X_train.columns if not col.startswith('review_')]
    
    logger.info(f"📈 Feature Analysis:")
    logger.info(f"   Baseline features: {len(baseline_features)}")
    logger.info(f"   New review features: {len(review_features)}")
    logger.info(f"   Total features: {len(X_train.columns)}")
    
    # Show new review features
    logger.info(f"\n🆕 New Review Features Added:")
    for feature in review_features[:10]:  # Show first 10
        logger.info(f"   • {feature}")
    if len(review_features) > 10:
        logger.info(f"   ... and {len(review_features) - 10} more")
    
    return review_features, baseline_features

def train_expanded_models(datasets):
    """Train models with expanded feature set."""
    logger = logging.getLogger(__name__)
    
    X_train = datasets['X_train']
    y_train = datasets['y_train'].iloc[:, 0]  # Get first column (target)
    X_val = datasets['X_val']
    y_val = datasets['y_val'].iloc[:, 0]
    X_test = datasets['X_test']
    y_test = datasets['y_test'].iloc[:, 0]
    
    # Handle any missing values
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)
    
    models = {
        'xgboost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        ),
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'logistic_regression': LogisticRegression(
            random_state=42,
            max_iter=1000
        )
    }
    
    results = {}
    trained_models = {}
    
    logger.info(f"🚀 Training models with expanded features...")
    
    for model_name, model in models.items():
        logger.info(f"\n📊 Training {model_name}...")
        start_time = time.time()
        
        try:
            # Train model
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            train_pred = model.predict(X_train)
            train_pred_proba = model.predict_proba(X_train)[:, 1]
            
            val_pred = model.predict(X_val)
            val_pred_proba = model.predict_proba(X_val)[:, 1]
            
            test_pred = model.predict(X_test)
            test_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            train_metrics = calculate_metrics(y_train, train_pred, train_pred_proba)
            val_metrics = calculate_metrics(y_val, val_pred, val_pred_proba)
            test_metrics = calculate_metrics(y_test, test_pred, test_pred_proba)
            
            # Cross-validation
            logger.info(f"   Running cross-validation...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            # Feature importance
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_names = X_train.columns.tolist()
                importance_scores = model.feature_importances_
                feature_importance = list(zip(feature_names, importance_scores))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Store results
            results[model_name] = {
                'model_name': model_name,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'training_time': training_time,
                'cross_validation': {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist()
                },
                'feature_importance': feature_importance[:20] if feature_importance else None  # Top 20
            }
            
            trained_models[model_name] = model
            
            logger.info(f"   ✅ {model_name} completed:")
            logger.info(f"      Test AUC: {test_metrics['roc_auc']:.4f}")
            logger.info(f"      CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            logger.info(f"      Training time: {training_time:.2f}s")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to train {model_name}: {str(e)}")
            continue
    
    return results, trained_models

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'samples': len(y_true),
        'default_rate': y_true.mean()
    }

def compare_with_baseline(results):
    """Compare expanded model results with baseline."""
    logger = logging.getLogger(__name__)
    
    # Load baseline results
    baseline_file = Path('results/model_training/training_results.json')
    if not baseline_file.exists():
        logger.warning("⚠️  Baseline results not found for comparison")
        return
    
    try:
        with open(baseline_file, 'r') as f:
            baseline_results = json.load(f)
        
        logger.info(f"\n📊 Performance Comparison (Baseline vs Expanded):")
        logger.info(f"{'Model':<20} {'Baseline AUC':<12} {'Expanded AUC':<12} {'Improvement':<12}")
        logger.info(f"{'-'*60}")
        
        improvements = {}
        
        for model_name in results.keys():
            if model_name in baseline_results:
                baseline_auc = baseline_results[model_name]['test_metrics']['roc_auc']
                expanded_auc = results[model_name]['test_metrics']['roc_auc']
                improvement = expanded_auc - baseline_auc
                improvement_pct = (improvement / baseline_auc) * 100
                
                improvements[model_name] = {
                    'baseline_auc': baseline_auc,
                    'expanded_auc': expanded_auc,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct
                }
                
                logger.info(f"{model_name:<20} {baseline_auc:<12.4f} {expanded_auc:<12.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        # Overall summary
        if improvements:
            avg_improvement = np.mean([imp['improvement'] for imp in improvements.values()])
            avg_improvement_pct = np.mean([imp['improvement_pct'] for imp in improvements.values()])
            
            logger.info(f"\n🎯 Overall Impact:")
            logger.info(f"   Average AUC improvement: {avg_improvement:+.4f} ({avg_improvement_pct:+.1f}%)")
            
            best_model = max(improvements.keys(), key=lambda k: improvements[k]['expanded_auc'])
            best_auc = improvements[best_model]['expanded_auc']
            logger.info(f"   Best performing model: {best_model} (AUC: {best_auc:.4f})")
        
        return improvements
        
    except Exception as e:
        logger.error(f"❌ Failed to load baseline results: {str(e)}")
        return None

def analyze_new_feature_importance(results):
    """Analyze importance of newly added review features."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"\n🔍 New Feature Importance Analysis:")
    
    for model_name, result in results.items():
        if result.get('feature_importance'):
            logger.info(f"\n📊 {model_name} - Top Review Features:")
            
            review_features = [
                (feature, importance) 
                for feature, importance in result['feature_importance']
                if feature.startswith('review_')
            ]
            
            for i, (feature, importance) in enumerate(review_features[:10]):
                logger.info(f"   {i+1:2d}. {feature:<30} {importance:.4f}")
            
            # Calculate total importance of review features
            total_importance = sum(imp for _, imp in result['feature_importance'])
            review_importance = sum(imp for _, imp in review_features)
            review_pct = (review_importance / total_importance) * 100
            
            logger.info(f"   📈 Review features contribute {review_pct:.1f}% of total importance")

def save_expanded_results(results, improvements=None):
    """Save expanded model results."""
    logger = logging.getLogger(__name__)
    
    # Save results
    results_dir = Path('results/model_training_expanded')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = results_dir / 'expanded_training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save comparison if available
    if improvements:
        comparison_file = results_dir / 'baseline_comparison.json'
        with open(comparison_file, 'w') as f:
            json.dump(improvements, f, indent=2)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_features': 66,
        'new_features': 28,
        'models_trained': len(results),
        'best_model': max(results.keys(), key=lambda k: results[k]['test_metrics']['roc_auc']),
        'best_auc': max(result['test_metrics']['roc_auc'] for result in results.values())
    }
    
    summary_file = results_dir / 'training_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"💾 Results saved to: {results_dir}")
    
    return results_file

def main():
    """Main execution function."""
    logger = setup_logging()
    
    print("🚀 Training Models with Expanded Free Review Features")
    print("=" * 60)
    
    # Load expanded data
    datasets = load_expanded_data()
    if datasets is None:
        return
    
    # Analyze features
    review_features, baseline_features = analyze_feature_importance(datasets)
    
    # Train models
    results, trained_models = train_expanded_models(datasets)
    
    if not results:
        logger.error("❌ No models trained successfully")
        return
    
    # Compare with baseline
    improvements = compare_with_baseline(results)
    
    # Analyze new feature importance
    analyze_new_feature_importance(results)
    
    # Save results
    results_file = save_expanded_results(results, improvements)
    
    # Final summary
    best_model = max(results.keys(), key=lambda k: results[k]['test_metrics']['roc_auc'])
    best_auc = results[best_model]['test_metrics']['roc_auc']
    
    print(f"\n🎉 Training Complete!")
    print(f"   Best Model: {best_model}")
    print(f"   Best AUC: {best_auc:.4f}")
    print(f"   Features: 38 → 66 (+{len(review_features)} review features)")
    print(f"   Results: {results_file}")
    
    if improvements:
        avg_improvement = np.mean([imp['improvement_pct'] for imp in improvements.values()])
        print(f"   Avg Improvement: {avg_improvement:+.1f}%")
        
        if avg_improvement > 1.0:
            print(f"   🎯 Target achieved! AUC improved by {avg_improvement:.1f}%")
        else:
            print(f"   📈 Good progress! Continue with more data sources for bigger gains")

if __name__ == "__main__":
    main()