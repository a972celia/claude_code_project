#!/usr/bin/env python3
"""
Train Enhanced Underwriting Model with Yelp Integration.
Demonstrates how external data integration improves model performance.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from models.model_trainer import UnderwritingModelTrainer

def setup_logging():
    """Configure logging for enhanced model training."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('enhanced_model_training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_enhanced_data():
    """Load enhanced preprocessed data with Yelp features."""
    logger = logging.getLogger(__name__)
    
    # Check if enhanced data exists
    enhanced_dir = Path("data/processed_enhanced")
    if not enhanced_dir.exists():
        logger.error("❌ Enhanced processed data not found. Run run_enhanced_preprocessing.py first.")
        return None
    
    logger.info("📂 Loading enhanced preprocessed data...")
    
    try:
        X_train = pd.read_parquet(enhanced_dir / "X_train_enhanced.parquet")
        X_val = pd.read_parquet(enhanced_dir / "X_val_enhanced.parquet") 
        X_test = pd.read_parquet(enhanced_dir / "X_test_enhanced.parquet")
        y_train = pd.read_parquet(enhanced_dir / "y_train_enhanced.parquet").iloc[:, 0]
        y_val = pd.read_parquet(enhanced_dir / "y_val_enhanced.parquet").iloc[:, 0]
        y_test = pd.read_parquet(enhanced_dir / "y_test_enhanced.parquet").iloc[:, 0]
        
        logger.info(f"✅ Enhanced data loaded:")
        logger.info(f"   Train: {X_train.shape}")
        logger.info(f"   Val: {X_val.shape}")
        logger.info(f"   Test: {X_test.shape}")
        
        # Check for Yelp features
        yelp_features = [col for col in X_train.columns if col.startswith('yelp_')]
        logger.info(f"   Yelp features: {len(yelp_features)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        logger.error(f"❌ Failed to load enhanced data: {str(e)}")
        return None

def load_baseline_data():
    """Load baseline data for comparison."""
    logger = logging.getLogger(__name__)
    
    try:
        # Load original processed data
        baseline_dir = Path("data/processed")
        
        X_train_base = pd.read_parquet(baseline_dir / "X_train.parquet")
        X_val_base = pd.read_parquet(baseline_dir / "X_val.parquet")
        X_test_base = pd.read_parquet(baseline_dir / "X_test.parquet")
        y_train_base = pd.read_parquet(baseline_dir / "y_train.parquet").iloc[:, 0]
        y_val_base = pd.read_parquet(baseline_dir / "y_val.parquet").iloc[:, 0]
        y_test_base = pd.read_parquet(baseline_dir / "y_test.parquet").iloc[:, 0]
        
        logger.info(f"📊 Baseline data loaded: {X_train_base.shape}")
        
        return X_train_base, X_val_base, X_test_base, y_train_base, y_val_base, y_test_base
        
    except Exception as e:
        logger.error(f"❌ Failed to load baseline data: {str(e)}")
        return None

def train_enhanced_model(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train model with enhanced features."""
    logger = logging.getLogger(__name__)
    logger.info("🤖 Training enhanced model with Yelp features...")
    
    # Initialize trainer for enhanced model
    trainer = UnderwritingModelTrainer(
        models_dir="models/enhanced_trained",
        results_dir="results/enhanced_model_training",
        random_state=42
    )
    
    # Store data in trainer format expected by the class
    trainer.feature_names = X_train.columns.tolist()
    
    # Train only XGBoost for demonstration (fastest)
    models = {'xgboost': trainer.get_baseline_models()['xgboost']}
    
    enhanced_results = {}
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name} with enhanced features...")
        start_time = trainer.time.time() if hasattr(trainer, 'time') else 0
        
        # Train model
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Evaluate
        results = trainer.evaluate_model(
            model, X_train, X_val, X_test,
            y_train, y_val, y_test, f"enhanced_{model_name}"
        )
        
        # Cross-validation
        cv_results = trainer.cross_validate_model(model, X_train, y_train, cv_folds=3)
        results['cross_validation'] = cv_results
        
        enhanced_results[f"enhanced_{model_name}"] = results
        
        logger.info(f"✅ Enhanced {model_name}:")
        logger.info(f"   Val AUC: {results['val_metrics']['roc_auc']:.4f}")
        logger.info(f"   Test AUC: {results['test_metrics']['roc_auc']:.4f}")
        logger.info(f"   CV AUC: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
    
    return enhanced_results

def train_baseline_model(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train baseline model for comparison."""
    logger = logging.getLogger(__name__)
    logger.info("📊 Training baseline model for comparison...")
    
    # Initialize trainer for baseline model
    trainer = UnderwritingModelTrainer(
        models_dir="models/baseline_comparison",
        results_dir="results/baseline_comparison",
        random_state=42
    )
    
    trainer.feature_names = X_train.columns.tolist()
    
    # Train only XGBoost for comparison
    model = trainer.get_baseline_models()['xgboost']
    
    logger.info("Training baseline xgboost...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Evaluate
    results = trainer.evaluate_model(
        model, X_train, X_val, X_test,
        y_train, y_val, y_test, "baseline_xgboost"
    )
    
    # Cross-validation
    cv_results = trainer.cross_validate_model(model, X_train, y_train, cv_folds=3)
    results['cross_validation'] = cv_results
    
    logger.info(f"✅ Baseline XGBoost:")
    logger.info(f"   Val AUC: {results['val_metrics']['roc_auc']:.4f}")
    logger.info(f"   Test AUC: {results['test_metrics']['roc_auc']:.4f}")
    logger.info(f"   CV AUC: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
    
    return {"baseline_xgboost": results}

def compare_models(enhanced_results, baseline_results):
    """Compare enhanced vs baseline model performance."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n🏆 MODEL PERFORMANCE COMPARISON")
    logger.info("=" * 60)
    
    # Extract key metrics
    enhanced_key = list(enhanced_results.keys())[0]
    baseline_key = list(baseline_results.keys())[0]
    
    enhanced_metrics = enhanced_results[enhanced_key]
    baseline_metrics = baseline_results[baseline_key]
    
    # Compare test AUC (most important metric)
    enhanced_auc = enhanced_metrics['test_metrics']['roc_auc']
    baseline_auc = baseline_metrics['test_metrics']['roc_auc']
    auc_improvement = enhanced_auc - baseline_auc
    
    logger.info(f"📊 Test AUC Comparison:")
    logger.info(f"   Baseline:  {baseline_auc:.4f}")
    logger.info(f"   Enhanced:  {enhanced_auc:.4f}")
    logger.info(f"   Improvement: {auc_improvement:+.4f} ({auc_improvement/baseline_auc*100:+.2f}%)")
    
    # Compare other metrics
    metrics = ['precision', 'recall', 'f1_score']
    logger.info(f"\n📈 Additional Metrics Comparison:")
    
    for metric in metrics:
        baseline_val = baseline_metrics['test_metrics'][metric]
        enhanced_val = enhanced_metrics['test_metrics'][metric]
        improvement = enhanced_val - baseline_val
        
        logger.info(f"   {metric.capitalize()}:")
        logger.info(f"     Baseline: {baseline_val:.4f}")
        logger.info(f"     Enhanced: {enhanced_val:.4f}")
        logger.info(f"     Change: {improvement:+.4f}")
    
    # Feature importance comparison (if available)
    if 'feature_importance' in enhanced_metrics:
        logger.info(f"\n🎯 Top Enhanced Features:")
        for i, feat in enumerate(enhanced_metrics['feature_importance'][:10], 1):
            marker = "🟡" if feat['feature'].startswith('yelp_') else "  "
            logger.info(f"   {i:2d}. {marker} {feat['feature']:<25} {feat['importance']:.4f}")
    
    # Summary
    if auc_improvement > 0:
        logger.info(f"\n✅ RESULT: Enhanced model with Yelp integration shows {auc_improvement*100:.2f} percentage point improvement in AUC!")
    else:
        logger.info(f"\n⚠️ RESULT: Enhanced model shows {abs(auc_improvement)*100:.2f} percentage point decrease in AUC.")
    
    return auc_improvement

def save_comparison_results(enhanced_results, baseline_results, auc_improvement):
    """Save model comparison results."""
    logger = logging.getLogger(__name__)
    
    comparison_dir = Path("results/yelp_integration_analysis")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    all_results = {
        'enhanced_models': enhanced_results,
        'baseline_models': baseline_results,
        'auc_improvement': float(auc_improvement),
        'analysis_date': pd.Timestamp.now().isoformat(),
        'conclusion': f"Yelp integration {'improved' if auc_improvement > 0 else 'did not improve'} model performance by {abs(auc_improvement)*100:.3f} percentage points"
    }
    
    with open(comparison_dir / "yelp_integration_analysis.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"💾 Analysis results saved to: {comparison_dir}")

def main():
    """Execute enhanced model training and comparison."""
    logger = setup_logging()
    logger.info("🌟 Starting Enhanced Model Training with Yelp Integration")
    
    try:
        # Load enhanced data
        enhanced_data = load_enhanced_data()
        if enhanced_data is None:
            logger.info("💡 To run this analysis, first run:")
            logger.info("   python scripts/external_data/run_enhanced_preprocessing.py")
            return
        
        X_train_enh, X_val_enh, X_test_enh, y_train_enh, y_val_enh, y_test_enh = enhanced_data
        
        # Load baseline data for comparison
        baseline_data = load_baseline_data()
        if baseline_data is None:
            logger.error("❌ Cannot load baseline data for comparison")
            return
        
        X_train_base, X_val_base, X_test_base, y_train_base, y_val_base, y_test_base = baseline_data
        
        # Train enhanced model
        enhanced_results = train_enhanced_model(
            X_train_enh, X_val_enh, X_test_enh, 
            y_train_enh, y_val_enh, y_test_enh
        )
        
        # Train baseline model
        baseline_results = train_baseline_model(
            X_train_base, X_val_base, X_test_base,
            y_train_base, y_val_base, y_test_base
        )
        
        # Compare results
        auc_improvement = compare_models(enhanced_results, baseline_results)
        
        # Save results
        save_comparison_results(enhanced_results, baseline_results, auc_improvement)
        
        logger.info("\n🎉 Enhanced model training and analysis completed!")
        
    except Exception as e:
        logger.error(f"❌ Enhanced model training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()