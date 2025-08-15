#!/usr/bin/env python3
"""
Display model training results for the AI-Powered Underwriting Engine.
"""

import pandas as pd
import json
from pathlib import Path

def main():
    """Display training results."""
    
    print("🤖 AI-POWERED UNDERWRITING ENGINE - MODEL TRAINING RESULTS")
    print("=" * 70)
    
    # Load model comparison
    comparison_path = Path("results/model_training/model_comparison.csv")
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        print("\n📊 MODEL PERFORMANCE COMPARISON:")
        print("-" * 50)
        print(f"{'Model':<20} {'Val AUC':<8} {'Test AUC':<8} {'CV AUC':<12} {'Time(s)':<8}")
        print("-" * 50)
        
        for _, row in df.iterrows():
            cv_display = f"{row['cv_auc_mean']:.3f}±{row['cv_auc_std']:.3f}"
            print(f"{row['model']:<20} {row['val_auc']:<8.4f} {row['test_auc']:<8.4f} {cv_display:<12} {row['training_time']:<8.1f}")
    
    # Load detailed results
    results_path = Path("results/model_training/training_results.json")
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Find best model
        best_model = None
        best_auc = 0
        for model_name, model_results in results.items():
            val_auc = model_results['val_metrics']['roc_auc']
            if val_auc > best_auc:
                best_auc = val_auc
                best_model = model_name
        
        if best_model:
            print(f"\n🏆 BEST MODEL: {best_model.upper()}")
            print("-" * 50)
            model_results = results[best_model]
            print(f"Validation AUC: {model_results['val_metrics']['roc_auc']:.4f}")
            print(f"Test AUC:       {model_results['test_metrics']['roc_auc']:.4f}")
            print(f"Precision:      {model_results['val_metrics']['precision']:.4f}")
            print(f"Recall:         {model_results['val_metrics']['recall']:.4f}")
            print(f"F1-Score:       {model_results['val_metrics']['f1_score']:.4f}")
            
            # Feature importance
            if 'feature_importance' in model_results and model_results['feature_importance']:
                print(f"\n🎯 TOP 10 FEATURES ({best_model.upper()}):")
                print("-" * 50)
                for i, feat in enumerate(model_results['feature_importance'][:10]):
                    print(f"{i+1:2d}. {feat['feature']:<25} {feat['importance']:.4f}")
            
            # SHAP analysis
            if 'shap_analysis' in model_results:
                shap_data = model_results['shap_analysis']
                print(f"\n🔍 SHAP EXPLAINABILITY ANALYSIS:")
                print("-" * 50)
                print(f"Explanation samples: {shap_data['explanation_samples']}")
                if 'feature_importance' in shap_data:
                    print("\nTop 10 SHAP Features:")
                    for i, feat in enumerate(shap_data['feature_importance'][:10]):
                        print(f"{i+1:2d}. {feat['feature']:<25} {feat['importance']:.4f}")
    
    print(f"\n💾 SAVED ARTIFACTS:")
    print("-" * 50)
    print("Models:      models/trained/")
    print("Results:     results/model_training/")
    print("Comparison:  results/model_training/model_comparison.csv")
    print("Full Data:   results/model_training/training_results.json")
    
    print(f"\n🎉 Model training completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()