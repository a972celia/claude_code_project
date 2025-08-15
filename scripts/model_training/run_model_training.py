#!/usr/bin/env python3
"""
Execute model training pipeline for the AI-Powered Underwriting Engine.
Trains baseline models and generates comprehensive performance reports.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from models.model_trainer import UnderwritingModelTrainer

def setup_logging():
    """Configure logging for training pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('model_training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Execute the complete model training pipeline."""
    logger = setup_logging()
    logger.info("🚀 Starting AI-Powered Underwriting Engine Model Training")
    
    try:
        # Initialize trainer
        trainer = UnderwritingModelTrainer(
            models_dir="models/trained",
            results_dir="results/model_training",
            random_state=42
        )
        
        # Load preprocessed data
        logger.info("📂 Loading preprocessed data...")
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()
        
        # Train baseline models
        logger.info("🤖 Training baseline models...")
        training_results = trainer.train_baseline_models(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Get best model and generate SHAP explanations
        logger.info("🎯 Generating model interpretability analysis...")
        best_name, best_model, best_results = trainer.get_best_model()
        
        # Generate SHAP explanations for best model
        shap_results = trainer.generate_shap_explanations(
            best_model, X_train.sample(n=1000, random_state=42), best_name
        )
        
        if 'error' not in shap_results:
            trainer.results[best_name]['shap_analysis'] = shap_results
            logger.info(f"✅ SHAP explanations generated for {best_name}")
        
        # Save models and results
        logger.info("💾 Saving models and results...")
        trainer.save_models_and_results()
        
        # Generate and display summary
        logger.info("📋 Generating training summary...")
        summary = trainer.generate_training_summary()
        print("\n" + "="*60)
        print(summary)
        print("="*60 + "\n")
        
        # Model-specific hyperparameter tuning for top 2 models
        logger.info("🔧 Performing hyperparameter tuning on best models...")
        
        # Get top 2 models by validation AUC
        sorted_models = sorted(
            training_results.items(),
            key=lambda x: x[1]['val_metrics']['roc_auc'],
            reverse=True
        )[:2]
        
        tuning_results = {}
        for model_name, _ in sorted_models:
            logger.info(f"Tuning {model_name}...")
            
            if model_name == 'xgboost':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.15]
                }
            elif model_name == 'lightgbm':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.15]
                }
            else:
                continue  # Skip tuning for other models
            
            tuned_model, tuning_result = trainer.hyperparameter_tuning(
                model_name, X_train, y_train, X_val, y_val, param_grid
            )
            
            # Evaluate tuned model
            tuned_results = trainer.evaluate_model(
                tuned_model, X_train, X_val, X_test,
                y_train, y_val, y_test, f"{model_name}_tuned"
            )
            
            # Add cross-validation for tuned model
            cv_results = trainer.cross_validate_model(tuned_model, X_train, y_train)
            tuned_results['cross_validation'] = cv_results
            
            # Store tuned model and results
            trainer.models[f"{model_name}_tuned"] = tuned_model
            trainer.results[f"{model_name}_tuned"] = tuned_results
            trainer.results[f"{model_name}_tuned"]['tuning_details'] = tuning_result
            tuning_results[f"{model_name}_tuned"] = tuned_results
        
        # Final summary with tuned models
        if tuning_results:
            logger.info("📊 Final model comparison with tuned models:")
            final_summary = trainer.generate_training_summary()
            print("\n" + "="*60)
            print("FINAL RESULTS (Including Tuned Models)")
            print("="*60)
            print(final_summary)
            print("="*60 + "\n")
            
            # Save updated results
            trainer.save_models_and_results()
        
        logger.info("🎉 Model training pipeline completed successfully!")
        
        # Final recommendations
        best_name, _, best_results = trainer.get_best_model()
        logger.info(f"🏆 Best performing model: {best_name}")
        logger.info(f"   Test AUC: {best_results['test_metrics']['roc_auc']:.4f}")
        logger.info(f"   Models saved to: models/trained/")
        logger.info(f"   Results saved to: results/model_training/")
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()