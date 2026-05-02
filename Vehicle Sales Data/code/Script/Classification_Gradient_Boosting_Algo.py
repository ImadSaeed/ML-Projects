"""
Training Gradient Boosting classification models
Models: XGBoost, LightGBM, CatBoost, AdaBoost
SAVES: Models (.pkl), Metrics (.csv), Classification Reports, Confusion Matrices
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import sys
import time
from datetime import datetime

# ============================================
# Add parent directory to Python path
# ============================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import preprocessing function for classification
from modules.preprocess import preprocess_vehicle_data_multiclass

# Import gradient boosting libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


class GradientBoostingClassification:
    """
    Class to train and evaluate Gradient Boosting classification models
    """
    
    def __init__(self, data_path):
        """
        Initialize with data path and load preprocessed data
        """
        print("="*60)
        print("GRADIENT BOOSTING CLASSIFICATION MODELS")
        print("="*60)
        
        # Load preprocessed data for multi-class classification
        print("\n\u1F4C2 Loading preprocessed data...")
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_vehicle_data_multiclass(data_path)
        
        print(f"\n\u2705 Data loaded successfully!")
        print(f"   Training set size: {self.X_train.shape}")
        print(f"   Test set size: {self.X_test.shape}")
        print(f"   Classes: 0=Low, 1=Medium, 2=High")
        
        # Create directories
        if 'kaggle' in sys.executable or 'kaggle' in os.getcwd():
            self.base_dir = '/kaggle/working'
            print(f"\n\u1F4C1 Running on Kaggle - saving to: {self.base_dir}")
        else:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            print(f"\n\u1F4C1 Running locally - saving to: {self.base_dir}")
        
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.results_dir = os.path.join(self.base_dir, 'results')
        self.evaluation_dir = os.path.join(self.base_dir, 'evaluation')
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
        print(f"   Models: {self.models_dir}")
        print(f"   Results: {self.results_dir}")
        print(f"   Evaluation: {self.evaluation_dir}")
        
        # Store results
        self.results = []
        self.class_names = ['Low', 'Medium', 'High']
    
    def _calculate_metrics(self, y_pred_train, y_pred_test, train_time, model_name):
        """
        Calculate classification metrics
        """
        # Training metrics
        train_acc = accuracy_score(self.y_train, y_pred_train)
        train_precision = precision_score(self.y_train, y_pred_train, average='weighted', zero_division=0)
        train_recall = recall_score(self.y_train, y_pred_train, average='weighted', zero_division=0)
        train_f1 = f1_score(self.y_train, y_pred_train, average='weighted', zero_division=0)
        
        # Testing metrics
        test_acc = accuracy_score(self.y_test, y_pred_test)
        test_precision = precision_score(self.y_test, y_pred_test, average='weighted', zero_division=0)
        test_recall = recall_score(self.y_test, y_pred_test, average='weighted', zero_division=0)
        test_f1 = f1_score(self.y_test, y_pred_test, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_test)
        
        metrics = {
            'train_acc': train_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'confusion_matrix': cm,
            'train_time': train_time
        }
        
        # Print metrics
        print(f"\n   [RESULTS]")
        print(f"   Training Accuracy: {train_acc:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Test Precision: {test_precision:.4f}")
        print(f"   Test Recall: {test_recall:.4f}")
        print(f"   Test F1-Score: {test_f1:.4f}")
        print(f"   Training time: {train_time:.2f} seconds")
        
        # Print confusion matrix
        print(f"\n   Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Low  Med  High")
        print(f"      Actual Low  {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
        print(f"            Med  {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
        print(f"            High {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")
        
        return metrics
    
    def _save_model(self, model, filename):
        """Save trained model"""
        filepath = os.path.join(self.models_dir, filename)
        joblib.dump(model, filepath)
        print(f"   [SAVE] Model saved: {filename}")
        
        # Save model info
        info_path = filepath.replace('.pkl', '_info.json')
        model_info = {
            'model_name': filename.replace('.pkl', ''),
            'saved_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': str(type(model).__name__),
            'file_path': filepath
        }
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
    
    def _save_classification_report(self, model_name, y_pred):
        """Save detailed classification report"""
        report = classification_report(self.y_test, y_pred, target_names=self.class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        filename = f"{model_name.lower().replace(' ', '_')}_classification_report.csv"
        filepath = os.path.join(self.evaluation_dir, filename)
        report_df.to_csv(filepath)
        print(f"   [FILE] Classification report saved: {filename}")
    
    def _save_predictions(self, model_name, y_pred):
        """Save predictions"""
        # Handle CatBoost returning 2D array (n_samples, 1)
        if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
        
        predictions_df = pd.DataFrame({
            'actual': self.y_test,
            'predicted': y_pred,
            'correct': self.y_test == y_pred
        })
        
        predictions_df['actual_class'] = predictions_df['actual'].map({0: 'Low', 1: 'Medium', 2: 'High'})
        predictions_df['predicted_class'] = predictions_df['predicted'].map({0: 'Low', 1: 'Medium', 2: 'High'})
        
        filename = f"{model_name.lower().replace(' ', '_')}_predictions.csv"
        filepath = os.path.join(self.evaluation_dir, filename)
        predictions_df.to_csv(filepath, index=False)
        print(f"   [FILE] Predictions saved: {filename}")
    
    def _store_results(self, model_name, metrics):
        """Store results for comparison"""
        self.results.append({
            'Model': model_name,
            'Train Accuracy': metrics['train_acc'],
            'Test Accuracy': metrics['test_acc'],
            'Test Precision': metrics['test_precision'],
            'Test Recall': metrics['test_recall'],
            'Test F1-Score': metrics['test_f1'],
            'Training Time (s)': metrics['train_time']
        })
    
    def _get_feature_importance(self, model, model_name):
        """Get and save feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        else:
            return
        
        feature_names = self.X_train.columns
        
        # Handle different importance formats
        if len(importances) != len(feature_names):
            # For LightGBM, importance might be array of arrays
            if hasattr(importances, 'shape') and len(importances.shape) > 1:
                importances = importances.flatten()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(10)
        
        print(f"\n   [TOP] Top 10 Features ({model_name}):")
        for idx, row in importance_df.iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
        
        # Save to file
        filename = f"{model_name.lower().replace(' ', '_')}_feature_importance.csv"
        filepath = os.path.join(self.evaluation_dir, filename)
        importance_df.to_csv(filepath, index=False)
        print(f"   [FILE] Feature importance saved: {filename}")
    
    def train_xgboost(self, n_estimators=200, learning_rate=0.05):
        """Train XGBoost Classifier"""
        print("\n" + "="*50)
        print(f"[TRAIN] XGBoost Classifier (n_estimators={n_estimators}, lr={learning_rate})")
        print("="*50)
        
        start_time = time.time()
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric='mlogloss'
        )
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time, 'XGBoost')
        self._save_model(model, 'xgboost_classifier.pkl')
        self._save_classification_report('XGBoost', y_pred_test)
        self._save_predictions('XGBoost', y_pred_test)
        self._get_feature_importance(model, 'XGBoost')
        self._store_results(f'XGBoost (n={n_estimators})', metrics)
        
        return model, metrics
    
    def train_lightgbm(self, n_estimators=200, learning_rate=0.05):
        """Train LightGBM Classifier"""
        print("\n" + "="*50)
        print(f"[TRAIN] LightGBM Classifier (n_estimators={n_estimators}, lr={learning_rate})")
        print("="*50)
        
        start_time = time.time()
        
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time, 'LightGBM')
        self._save_model(model, 'lightgbm_classifier.pkl')
        self._save_classification_report('LightGBM', y_pred_test)
        self._save_predictions('LightGBM', y_pred_test)
        self._get_feature_importance(model, 'LightGBM')
        self._store_results(f'LightGBM (n={n_estimators})', metrics)
        
        return model, metrics
    
    def train_catboost(self, iterations=200, learning_rate=0.05):
        """Train CatBoost Classifier"""
        print("\n" + "="*50)
        print(f"[TRAIN] CatBoost Classifier (iterations={iterations}, lr={learning_rate})")
        print("="*50)
        
        start_time = time.time()
        
        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=6,
            random_seed=42,
            verbose=False
        )
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Flatten predictions if needed (CatBoost returns 2D)
        if hasattr(y_pred_train, 'shape') and len(y_pred_train.shape) > 1:
            y_pred_train = y_pred_train.flatten()
        if hasattr(y_pred_test, 'shape') and len(y_pred_test.shape) > 1:
            y_pred_test = y_pred_test.flatten()
        
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time, 'CatBoost')
        self._save_model(model, 'catboost_classifier.pkl')
        self._save_classification_report('CatBoost', y_pred_test)
        self._save_predictions('CatBoost', y_pred_test)
        self._get_feature_importance(model, 'CatBoost')
        self._store_results(f'CatBoost (iter={iterations})', metrics)
        
        return model, metrics
    
    def train_adaboost(self, n_estimators=200, learning_rate=0.05):
        """Train AdaBoost Classifier"""
        print("\n" + "="*50)
        print(f"[TRAIN] AdaBoost Classifier (n_estimators={n_estimators}, lr={learning_rate})")
        print("="*50)
        
        start_time = time.time()
        
        model = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time, 'AdaBoost')
        self._save_model(model, 'adaboost_classifier.pkl')
        self._save_classification_report('AdaBoost', y_pred_test)
        self._save_predictions('AdaBoost', y_pred_test)
        self._get_feature_importance(model, 'AdaBoost')
        self._store_results(f'AdaBoost (n={n_estimators})', metrics)
        
        return model, metrics
    
    def save_all_results(self):
        """Save all results to CSV and JSON"""
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('Test Accuracy', ascending=False)
        
        csv_path = os.path.join(self.results_dir, 'gradient_boosting_classification_comparison.csv')
        results_df.to_csv(csv_path, index=False)
        
        json_path = os.path.join(self.results_dir, 'gradient_boosting_classification_comparison.json')
        results_df.to_json(json_path, orient='records', indent=4)
        
        print("\n" + "="*60)
        print("[RESULTS] GRADIENT BOOSTING CLASSIFICATION COMPARISON")
        print("="*60)
        print(results_df.to_string(index=False))
        
        print(f"\n[SAVE] Results saved to:")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")
        
        if len(results_df) > 0:
            best_model = results_df.iloc[0]
            print(f"\n[WINNER] BEST GRADIENT BOOSTING MODEL: {best_model['Model']}")
            print(f"   Test Accuracy: {best_model['Test Accuracy']:.4f}")
            print(f"   Test F1-Score: {best_model['Test F1-Score']:.4f}")
        
        return results_df
    
    def run_all_models(self):
        """Run all gradient boosting classification models"""
        print("\n" + "="*60)
        print("RUNNING ALL GRADIENT BOOSTING CLASSIFICATION MODELS")
        print("="*60)
        print("\n[INFO] Models to train: 4\n")
        
        # XGBoost
        self.train_xgboost(n_estimators=200, learning_rate=0.05)
        
        # LightGBM
        self.train_lightgbm(n_estimators=200, learning_rate=0.05)
        
        # CatBoost
        self.train_catboost(iterations=200, learning_rate=0.05)
        
        # AdaBoost
        self.train_adaboost(n_estimators=200, learning_rate=0.05)
        
        results_df = self.save_all_results()
        return results_df


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("GRADIENT BOOSTING CLASSIFICATION MODELS")
    print("="*60)
    
    # Your Kaggle path
    file_path = '/kaggle/input/datasets/imadsaeed123/vehicle-sale-data/Vehicle Sales Data/archive/car_prices.csv'
    
    try:
        trainer = GradientBoostingClassification(file_path)
        results = trainer.run_all_models()
        
        print("\n" + "="*60)
        print("[DONE] ALL GRADIENT BOOSTING MODELS TRAINED SUCCESSFULLY!")
        print("="*60)
        
        print("\n[FILES] SAVED:")
        print("   /models/")
        print("      ├── xgboost_classifier.pkl")
        print("      ├── lightgbm_classifier.pkl")
        print("      ├── catboost_classifier.pkl")
        print("      └── adaboost_classifier.pkl")
        
        print("\n   /results/")
        print("      └── gradient_boosting_classification_comparison.csv")
        
        print("\n   /evaluation/")
        print("      ├── *_predictions.csv (4 files)")
        print("      ├── *_classification_report.csv (4 files)")
        print("      └── *_feature_importance.csv (4 files)")
        
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {file_path}")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")