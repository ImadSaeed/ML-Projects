"""
Training multiple classification models using scikit-learn
Models: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Naive Bayes, Gradient Boosting Classifier
SAVES: Models (.pkl), Metrics (.csv), Training details (.json), Classification Reports
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import sys
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

# Classification models from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Metrics for evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# For timing
import time

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


class ClassificationModels:
    """
    Class to train and evaluate multiple classification models
    """
    
    def __init__(self, data_path):
        """
        Initialize with data path and load preprocessed data
        """
        print("="*60)
        print("CLASSIFICATION MODELS TRAINING")
        print("="*60)
        
        # Load preprocessed data for multi-class classification
        print("\u1F4C2 Loading preprocessed data...")
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_vehicle_data_multiclass(data_path)
        
        print(f"\u2705 Data loaded successfully!")
        print(f"   Training set size: {self.X_train.shape}")
        print(f"   Test set size: {self.X_test.shape}")
        print(f"   Classes: 0=Low, 1=Medium, 2=High")
        
        # Create directories
        if 'kaggle' in sys.executable or 'kaggle' in os.getcwd():
            self.base_dir = '/kaggle/working'
            print(f"\u1F4C1 Running on Kaggle - saving to: {self.base_dir}")
        else:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            print(f"\u1F4C1 Running locally - saving to: {self.base_dir}")
        
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
        print(f"\n\u1F4CA Results:")
        print(f"   Training Accuracy: {train_acc:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Test Precision: {test_precision:.4f}")
        print(f"   Test Recall: {test_recall:.4f}")
        print(f"   Test F1-Score: {test_f1:.4f}")
        print(f"   Training time: {train_time:.2f} seconds")
        
        # Print confusion matrix
        print(f"\n\u1F4CA Confusion Matrix:")
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
        print(f"\u1F4BE Model saved: {filename}")
        
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
        print(f"\u1F4DD Classification report saved: {filename}")
    
    def _save_predictions(self, model_name, y_pred):
        """Save predictions"""
        predictions_df = pd.DataFrame({
            'actual': self.y_test,
            'predicted': y_pred,
            'correct': self.y_test == y_pred
        })
        
        # Add actual and predicted class names
        predictions_df['actual_class'] = predictions_df['actual'].map({0: 'Low', 1: 'Medium', 2: 'High'})
        predictions_df['predicted_class'] = predictions_df['predicted'].map({0: 'Low', 1: 'Medium', 2: 'High'})
        
        filename = f"{model_name.lower().replace(' ', '_')}_predictions.csv"
        filepath = os.path.join(self.evaluation_dir, filename)
        predictions_df.to_csv(filepath, index=False)
        print(f"\u1F4DD Predictions saved: {filename}")
    
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
        """Get feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.X_train.columns
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(10)
            
            print(f"\n\u1F50D Top 10 Features ({model_name}):")
            for idx, row in importance_df.iterrows():
                print(f"      {row['feature']}: {row['importance']:.4f}")
            
            # Save to file
            filename = f"{model_name.lower().replace(' ', '_')}_feature_importance.csv"
            filepath = os.path.join(self.evaluation_dir, filename)
            importance_df.to_csv(filepath, index=False)
            print(f"\u1F4CA Feature importance saved: {filename}")
    
    def train_logistic_regression(self, max_iter=1000):
        """Train Logistic Regression"""
        print("\n" + "="*50)
        print(" Training Logistic Regression")
        print("="*50)
        
        start_time = time.time()
        
        model = LogisticRegression(
            max_iter=max_iter,
            random_state=42,
            n_jobs=-1,
            multi_class='ovr'
        )
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time, 'Logistic Regression')
        self._save_model(model, 'logistic_regression.pkl')
        self._save_classification_report('Logistic Regression', y_pred_test)
        self._save_predictions('Logistic Regression', y_pred_test)
        self._store_results('Logistic Regression', metrics)
        
        return model, metrics
    
    def train_knn_classifier(self, n_neighbors=5):
        """Train K-Neighbors Classifier"""
        print("\n" + "="*50)
        print(f" Training K-Neighbors Classifier (k={n_neighbors})")
        print("="*50)
        
        start_time = time.time()
        
        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time, 'KNN Classifier')
        self._save_model(model, 'knn_classifier.pkl')
        self._save_classification_report('KNN Classifier', y_pred_test)
        self._save_predictions('KNN Classifier', y_pred_test)
        self._store_results(f'KNN Classifier (k={n_neighbors})', metrics)
        
        return model, metrics
    
    def train_svm_classifier(self, kernel='rbf'):
        """Train SVM Classifier"""
        print("\n" + "="*50)
        print(f" Training SVM Classifier (kernel={kernel})")
        print("="*50)
        print("  SVM may be slow on large datasets. Using sample...")
        
        start_time = time.time()
        
        # Use sample for SVM if dataset is large
        use_sample = len(self.X_train) > 50000
        if use_sample:
            print("   Using 50,000 samples for SVM")
            X_train_sample = self.X_train[:50000]
            y_train_sample = self.y_train[:50000]
        else:
            X_train_sample = self.X_train
            y_train_sample = self.y_train
        
        model = SVC(kernel=kernel, random_state=42)
        model.fit(X_train_sample, y_train_sample)
        
        train_time = time.time() - start_time
        
        # Predict on full test set
        y_pred_test = model.predict(self.X_test)
        y_pred_train = model.predict(X_train_sample)
        
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time, 'SVM')
        self._save_model(model, 'svm_classifier.pkl')
        self._save_classification_report('SVM', y_pred_test)
        self._save_predictions('SVM', y_pred_test)
        self._store_results(f'SVM (kernel={kernel})', metrics)
        
        return model, metrics
    
    def train_decision_tree_classifier(self, max_depth=20):
        """Train Decision Tree Classifier"""
        print("\n" + "="*50)
        print(f" Training Decision Tree Classifier (max_depth={max_depth})")
        print("="*50)
        
        start_time = time.time()
        
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time, 'Decision Tree')
        self._save_model(model, 'decision_tree_classifier.pkl')
        self._save_classification_report('Decision Tree', y_pred_test)
        self._save_predictions('Decision Tree', y_pred_test)
        self._get_feature_importance(model, 'Decision Tree')
        self._store_results(f'Decision Tree (max_depth={max_depth})', metrics)
        
        return model, metrics
    
    def train_random_forest_classifier(self, n_estimators=100):
        """Train Random Forest Classifier"""
        print("\n" + "="*50)
        print(f" Training Random Forest Classifier (n_estimators={n_estimators})")
        print("="*50)
        
        start_time = time.time()
        
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time, 'Random Forest')
        self._save_model(model, 'random_forest_classifier.pkl')
        self._save_classification_report('Random Forest', y_pred_test)
        self._save_predictions('Random Forest', y_pred_test)
        self._get_feature_importance(model, 'Random Forest')
        self._store_results(f'Random Forest (n={n_estimators})', metrics)
        
        return model, metrics
    
    def train_naive_bayes(self):
        """Train Gaussian Naive Bayes"""
        print("\n" + "="*50)
        print(" Training Gaussian Naive Bayes")
        print("="*50)
        
        start_time = time.time()
        
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time, 'Naive Bayes')
        self._save_model(model, 'naive_bayes.pkl')
        self._save_classification_report('Naive Bayes', y_pred_test)
        self._save_predictions('Naive Bayes', y_pred_test)
        self._store_results('Naive Bayes', metrics)
        
        return model, metrics
    
    def train_gradient_boosting_classifier(self, n_estimators=100):
        """Train Gradient Boosting Classifier"""
        print("\n" + "="*50)
        print(f" Training Gradient Boosting Classifier (n_estimators={n_estimators})")
        print("="*50)
        
        start_time = time.time()
        
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time, 'Gradient Boosting')
        self._save_model(model, 'gradient_boosting_classifier.pkl')
        self._save_classification_report('Gradient Boosting', y_pred_test)
        self._save_predictions('Gradient Boosting', y_pred_test)
        self._get_feature_importance(model, 'Gradient Boosting')
        self._store_results(f'Gradient Boosting (n={n_estimators})', metrics)
        
        return model, metrics
    
    def save_all_results(self):
        """Save all results to CSV and JSON"""
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('Test Accuracy', ascending=False)
        
        csv_path = os.path.join(self.results_dir, 'classification_model_comparison.csv')
        results_df.to_csv(csv_path, index=False)
        
        json_path = os.path.join(self.results_dir, 'classification_model_comparison.json')
        results_df.to_json(json_path, orient='records', indent=4)
        
        print("\n" + "="*60)
        print("\u1F4CA CLASSIFICATION MODEL COMPARISON")
        print("="*60)
        print(results_df.to_string(index=False))
        
        print(f"\n Results saved to:")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")
        
        if len(results_df) > 0:
            best_model = results_df.iloc[0]
            print(f"\u1F3C5 BEST CLASSIFICATION MODEL: {best_model['Model']}")
            print(f"   Test Accuracy: {best_model['Test Accuracy']:.4f}")
            print(f"   Test F1-Score: {best_model['Test F1-Score']:.4f}")
        
        return results_df
    
    def run_all_models(self):
        """
        Run all classification models
        """
        print("\n" + "\uD83D\uDD25"*10)
        print("RUNNING ALL CLASSIFICATION MODELS")
        print("\uD83D\uDD25"*10)
        print("\n\u1F4DD Models to train: 8\n")
        
        # Logistic Regression
        self.train_logistic_regression()
        
        # KNN Classifier
        self.train_knn_classifier(n_neighbors=5)
        
        # SVM (optional - may be slow)
        # self.train_svm_classifier(kernel='rbf')
        
        # Decision Tree
        self.train_decision_tree_classifier(max_depth=20)
        
        # Random Forest
        self.train_random_forest_classifier(n_estimators=100)
        
        # Naive Bayes
        self.train_naive_bayes()
        
        # Gradient Boosting Classifier
        self.train_gradient_boosting_classifier(n_estimators=100)
        
        results_df = self.save_all_results()
        return results_df


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("\u1F3E0 VEHICLE SALES PRICE CLASSIFICATION - SKLEARN MODELS")
    print("="*60)
    
    # Your Kaggle path
    file_path = '/kaggle/input/datasets/imadsaeed123/vehicle-sale-data/Vehicle Sales Data/archive/car_prices.csv'
    
    try:
        trainer = ClassificationModels(file_path)
        results = trainer.run_all_models()
        
        print("\n" + "="*60)
        print("\u2705 ALL CLASSIFICATION MODELS TRAINED SUCCESSFULLY!")
        print("="*60)
        
        print("\n\u1F4C1 FILES SAVED:")
        print("   \u1F4C2 /models/ (.pkl files)")
        print("      ├── logistic_regression.pkl")
        print("      ├── knn_classifier.pkl")
        print("      ├── decision_tree_classifier.pkl")
        print("      ├── random_forest_classifier.pkl")
        print("      ├── naive_bayes.pkl")
        print("      └── gradient_boosting_classifier.pkl")
        
        print("\n\u1F4C1 /results/")
        print("      └── classification_model_comparison.csv")
        
        print("\n\u1F4CB /evaluation/")
        print("      \u2B07── *_predictions.csv")
        print("      \u2B07── *_classification_report.csv")
        print("      \u2B07── *_feature_importance.csv (for tree models)")
        
        
    except FileNotFoundError:
        print(f"\n\u274C File not found: {file_path}")
        
    except Exception as e:
        print(f"\n\u274C An error occurred: {e}")