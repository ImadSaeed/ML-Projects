"""
Training multiple regression models using scikit-learn
Models: Linear Regression, Ridge, Lasso, ElasticNet, Bayesian Ridge, KNN, Decision Tree, Random Forest
SAVES: Models (.pkl), Metrics (.csv), Training details (.json)
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import sys
from datetime import datetime

# ============================================
# Add parent directory to Python path so 'modules' can be found
# ============================================
current_dir = os.path.dirname(os.path.abspath(__file__))  # /code/Script/
parent_dir = os.path.dirname(current_dir)                # /code/
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# ============================================

# Import preprocessing function
from modules.preprocess import preprocess_vehicle_data_regression

# Regression models from sklearn
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Metrics for evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# For timing
import time

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


class RegressionModels:
    """
    Class to train and evaluate multiple regression models
    """
    
    def __init__(self, data_path):
        """
        Initialize with data path and load preprocessed data
        """
        print("="*60)
        print("REGRESSION MODELS TRAINING")
        print("="*60)
        
        # Load preprocessed data
        print("\n\u1F4C2 Loading preprocessed data...")
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_vehicle_data_regression(data_path)
        
        print(f"\n\u2705 Data loaded successfully!")
        print(f"   Training set size: {self.X_train.shape}")
        print(f"   Test set size: {self.X_test.shape}")
        
        # ============================================
        # Save to /kaggle/working/ on Kaggle
        # ============================================
        if 'kaggle' in sys.executable or 'kaggle' in os.getcwd():
            # On Kaggle - save to working directory
            self.base_dir = '/kaggle/working'
            print(f"\n\u1F4C1 Running on Kaggle - saving to: {self.base_dir}")
        else:
            # Local - save relative to project
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            print(f"\n\u1F4C1 Running locally - saving to: {self.base_dir}")
        
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.results_dir = os.path.join(self.base_dir, 'results')
        self.evaluation_dir = os.path.join(self.base_dir, 'evaluation')
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
        print(f"   Models will be saved to: {self.models_dir}")
        print(f"   Results will be saved to: {self.results_dir}")
        print(f"   Evaluation files to: {self.evaluation_dir}")
        
        # Store results
        self.results = []
        self.all_predictions = {}  # Store predictions for later visualization
    
    def train_linear_regression(self):
        """Train Linear Regression model"""
        print("\n" + "="*50)
        print(" Training Linear Regression")
        print("="*50)
        
        start_time = time.time()
        
        # Create and train model
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Store predictions for later
        self.all_predictions['Linear Regression'] = {
            'y_true': self.y_test,
            'y_pred': y_pred_test
        }
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time)
        
        # Save model
        self._save_model(model, 'linear_regression.pkl')
        
        # Save predictions
        self._save_predictions('Linear Regression', y_pred_test)
        
        # Store results
        self._store_results('Linear Regression', metrics)
        
        return model, metrics
    
    def train_knn_regressor(self, n_neighbors=5):
        """Train K-Neighbors Regressor"""
        print("\n" + "="*50)
        print(f" Training K-Neighbors Regressor (k={n_neighbors})")
        print("="*50)
        
        start_time = time.time()
        
        # Create and train model
        model = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Store predictions
        self.all_predictions['KNN Regressor'] = {
            'y_true': self.y_test,
            'y_pred': y_pred_test
        }
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time)
        
        # Save model
        self._save_model(model, 'knn_regressor.pkl')
        
        # Save predictions
        self._save_predictions('KNN Regressor', y_pred_test)
        
        # Store results
        self._store_results(f'KNN Regressor (k={n_neighbors})', metrics)
        
        return model, metrics
    
    def train_decision_tree(self, max_depth=20):
        """Train Decision Tree Regressor"""
        print("\n" + "="*50)
        print(f" Training Decision Tree Regressor (max_depth={max_depth})")
        print("="*50)
        
        start_time = time.time()
        
        # Create and train model
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Store predictions
        self.all_predictions['Decision Tree'] = {
            'y_true': self.y_test,
            'y_pred': y_pred_test
        }
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time)
        
        # Save model
        self._save_model(model, 'decision_tree_regressor.pkl')
        
        # Save predictions
        self._save_predictions('Decision Tree', y_pred_test)
        
        # Save feature importance
        self._save_feature_importance(model, 'decision_tree')
        
        # Store results
        self._store_results(f'Decision Tree (max_depth={max_depth})', metrics)
        
        return model, metrics
    
    def train_random_forest(self, n_estimators=100):
        """Train Random Forest Regressor"""
        print("\n" + "="*50)
        print(f" Training Random Forest Regressor (n_estimators={n_estimators})")
        print("="*50)
        
        start_time = time.time()
        
        # Create and train model
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Store predictions
        self.all_predictions['Random Forest'] = {
            'y_true': self.y_test,
            'y_pred': y_pred_test
        }
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time)
        
        # Save model
        self._save_model(model, 'random_forest_regressor.pkl')
        
        # Save predictions
        self._save_predictions('Random Forest', y_pred_test)
        
        # Save feature importance
        self._save_feature_importance(model, 'random_forest')
        
        # Store results
        self._store_results(f'Random Forest (n={n_estimators})', metrics)
        
        return model, metrics
    
    def train_bayesian_ridge(self):
        """Train Bayesian Ridge Regressor"""
        print("\n" + "="*50)
        print(" Training Bayesian Ridge Regressor")
        print("="*50)
        
        start_time = time.time()
        
        # Create and train model
        model = BayesianRidge()
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Store predictions
        self.all_predictions['Bayesian Ridge'] = {
            'y_true': self.y_test,
            'y_pred': y_pred_test
        }
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time)
        
        # Save model
        self._save_model(model, 'bayesian_ridge.pkl')
        
        # Save predictions
        self._save_predictions('Bayesian Ridge', y_pred_test)
        
        # Store results
        self._store_results('Bayesian Ridge', metrics)
        
        return model, metrics
    
    def train_ridge(self, alpha=1.0):
        """Train Ridge Regression (L2 regularization)"""
        print("\n" + "="*50)
        print(f" Training Ridge Regression (alpha={alpha})")
        print("="*50)
        
        start_time = time.time()
        
        # Create and train model
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Store predictions
        self.all_predictions['Ridge'] = {
            'y_true': self.y_test,
            'y_pred': y_pred_test
        }
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time)
        
        # Save model
        self._save_model(model, 'ridge_regression.pkl')
        
        # Save predictions
        self._save_predictions('Ridge', y_pred_test)
        
        # Store results
        self._store_results(f'Ridge Regression (alpha={alpha})', metrics)
        
        return model, metrics
    
    def train_lasso(self, alpha=1.0):
        """Train Lasso Regression (L1 regularization)"""
        print("\n" + "="*50)
        print(f"📈 Training Lasso Regression (alpha={alpha})")
        print("="*50)
        
        start_time = time.time()
        
        # Create and train model
        model = Lasso(alpha=alpha, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Store predictions
        self.all_predictions['Lasso'] = {
            'y_true': self.y_test,
            'y_pred': y_pred_test
        }
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time)
        
        # Save model
        self._save_model(model, 'lasso_regression.pkl')
        
        # Save predictions
        self._save_predictions('Lasso', y_pred_test)
        
        # Store results
        self._store_results(f'Lasso Regression (alpha={alpha})', metrics)
        
        return model, metrics
    
    def train_elasticnet(self, alpha=1.0, l1_ratio=0.5):
        """Train ElasticNet Regression"""
        print("\n" + "="*50)
        print(f" Training ElasticNet (alpha={alpha}, l1_ratio={l1_ratio})")
        print("="*50)
        
        start_time = time.time()
        
        # Create and train model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Store predictions
        self.all_predictions['ElasticNet'] = {
            'y_true': self.y_test,
            'y_pred': y_pred_test
        }
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time)
        
        # Save model
        self._save_model(model, 'elasticnet_regression.pkl')
        
        # Save predictions
        self._save_predictions('ElasticNet', y_pred_test)
        
        # Store results
        self._store_results(f'ElasticNet (alpha={alpha})', metrics)
        
        return model, metrics
    
    def _calculate_metrics(self, y_pred_train, y_pred_test, train_time):
        """
        Calculate regression metrics
        """
        # Training metrics
        train_mse = mean_squared_error(self.y_train[:len(y_pred_train)], y_pred_train)
        train_mae = mean_absolute_error(self.y_train[:len(y_pred_train)], y_pred_train)
        train_r2 = r2_score(self.y_train[:len(y_pred_train)], y_pred_train)
        
        # Testing metrics
        test_mse = mean_squared_error(self.y_test, y_pred_test)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        test_r2 = r2_score(self.y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        y_test_actual = self._inverse_transform_price(self.y_test)
        y_pred_actual = self._inverse_transform_price(y_pred_test)
        mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
        
        metrics = {
            'train_mse': train_mse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mape': mape,
            'train_time': train_time
        }
        
        # Print metrics
        print(f"\n   \u1F4C8 Results:")
        print(f"   Training R\u00b2: {train_r2:.4f}")
        print(f"   Test R\u00b2: {test_r2:.4f}")
        print(f"   Test MAE: \u0024{self._inverse_transform_price(test_mae):,.2f}")
        print(f"   Test RMSE: \u0024{self._inverse_transform_price(test_rmse):,.2f}")
        print(f"   Test MAPE: {mape:.2f}%")
        print(f"   Training time: {train_time:.2f} seconds")
        
        return metrics
    
    def _inverse_transform_price(self, log_value):
        """
        Convert log-transformed price back to actual price
        """
        return np.expm1(log_value)
    
    def _store_results(self, model_name, metrics):
        """
        Store results for later comparison
        """
        self.results.append({
            'Model': model_name,
            'Train R\u00b2': metrics['train_r2'],
            'Test R\u00b2': metrics['test_r2'],
            'Test MAE (\u0024)': self._inverse_transform_price(metrics['test_mae']),
            'Test RMSE (\u0024)': self._inverse_transform_price(metrics['test_rmse']),
            'Test MAPE (%)': metrics['test_mape'],
            'Training Time (s)': metrics['train_time']
        })
    
    def _save_model(self, model, filename):
        """
        Save trained model to models folder using joblib (.pkl format)
        """
        filepath = os.path.join(self.models_dir, filename)
        joblib.dump(model, filepath)
        print(f"   \u1F4BE Model saved: {filename}")
        
        # Also save model info as JSON
        info_path = filepath.replace('.pkl', '_info.json')
        model_info = {
            'model_name': filename.replace('.pkl', ''),
            'saved_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': str(type(model).__name__),
            'file_path': filepath
        }
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
    
    def _save_predictions(self, model_name, y_pred):
        """
        Save predictions for later visualization
        """
        predictions_df = pd.DataFrame({
            'actual_price_log': self.y_test,
            'predicted_price_log': y_pred,
            'actual_price': self._inverse_transform_price(self.y_test),
            'predicted_price': self._inverse_transform_price(y_pred),
            'residual': self.y_test - y_pred,
            'residual_percentage': ((self.y_test - y_pred) / self.y_test) * 100
        })
        
        filename = f"{model_name.lower().replace(' ', '_')}_predictions.csv"
        filepath = os.path.join(self.evaluation_dir, filename)
        predictions_df.to_csv(filepath, index=False)
        print(f"   \u1F4DD Predictions saved: {filename}")
    
    def _save_feature_importance(self, model, model_name):
        """
        Save feature importance for tree-based models
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.X_train.columns
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            filename = f"{model_name}_feature_importance.csv"
            filepath = os.path.join(self.evaluation_dir, filename)
            importance_df.to_csv(filepath, index=False)
            
            print(f"\n   \u1F4C8 Top 10 Most Important Features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"      {row['feature']}: {row['importance']:.4f}")
            
            print(f"   \u1F4CA Feature importance saved: {filename}")
    
    def save_all_results(self):
        """
        Save all results to CSV and JSON
        """
        # Save model comparison
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('Test R\u00b2', ascending=False)
        
        csv_path = os.path.join(self.results_dir, 'regression_model_comparison.csv')
        results_df.to_csv(csv_path, index=False)
        
        # Save as JSON as well
        json_path = os.path.join(self.results_dir, 'regression_model_comparison.json')
        results_df.to_json(json_path, orient='records', indent=4)
        
        print("\n" + "="*60)
        print(" MODEL COMPARISON SUMMARY")
        print("="*60)
        print(results_df.to_string(index=False))
        
        print(f"\n\u2705 RESULTS SAVED TO:")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")
        
        # Find best model
        best_model = results_df.iloc[0]
        print(f"\n\u1F3C6 BEST MODEL: {best_model['Model']}")
        print(f"   Test R\u00b2: {best_model['Test R\u00b2']:.4f}")
        print(f"   Test MAE: ${best_model['Test MAE (\u0024)']:,.2f}")
        
        return results_df
    
    def run_all_models(self):
        """
        Run all regression models 
        """
        print("\n" + "🔥"*30)
        print("RUNNING ALL REGRESSION MODELS")
        print("🔥"*30)
        print("\n\u1F4C1 Models to train: 8 models\n")
        
        # Linear models (5 models)
        self.train_linear_regression()
        self.train_ridge(alpha=1.0)
        self.train_lasso(alpha=1.0)
        self.train_elasticnet(alpha=1.0, l1_ratio=0.5)
        self.train_bayesian_ridge()
        
        # Tree-based models (2 models)
        self.train_decision_tree(max_depth=20)
        self.train_random_forest(n_estimators=100)
        
        # Distance-based model (1 model)
        self.train_knn_regressor(n_neighbors=5)
        
        # Save everything
        results_df = self.save_all_results()
        
        return results_df


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*60)
    print(" VEHICLE SALES PRICE PREDICTION - REGRESSION MODELS")
    print("="*60)
    
    # Your Kaggle path
    file_path = '/kaggle/input/datasets/imadsaeed123/vehicle-sale-data/Vehicle Sales Data/archive/car_prices.csv'
    
    # Alternative for local
    # file_path = '../../../data/vehicle_sales.csv'
    
    try:
        # Initialize trainer
        trainer = RegressionModels(file_path)
        
        # Run all models
        results = trainer.run_all_models()
        
        print("\n" + "="*60)
        print(" ALL REGRESSION MODELS TRAINED SUCCESSFULLY!")
        print("="*60)
        
        print("\n\u1F4C2 FILES SAVED:")
        print("\n   \u1F4C1 /models/ (Model files - .pkl format)")
        print("      ├── linear_regression.pkl")
        print("      ├── ridge_regression.pkl")
        print("      ├── lasso_regression.pkl")
        print("      ├── elasticnet_regression.pkl")
        print("      ├── bayesian_ridge.pkl")
        print("      ├── decision_tree_regressor.pkl")
        print("      ├── random_forest_regressor.pkl")
        print("      └── knn_regressor.pkl")
        
        print("\n   \u1F4C2 /results/ (Comparison results)")
        print("      ├── regression_model_comparison.csv")
        print("      └── regression_model_comparison.json")
        
        print("\n   \u1F4DD /evaluation/ (Predictions for visualization)")
        print("      ├── linear_regression_predictions.csv")
        print("      ├── ridge_predictions.csv")
        print("      ├── lasso_predictions.csv")
        print("      ├── elasticnet_predictions.csv")
        print("      ├── bayesian_ridge_predictions.csv")
        print("      ├── decision_tree_predictions.csv")
        print("      ├── decision_tree_feature_importance.csv")
        print("      ├── random_forest_predictions.csv")
        print("      ├── random_forest_feature_importance.csv")
        print("      └── knn_regressor_predictions.csv")
        
        print("\n\u1F4C8 MODELS TRAINED (8 total):")
        print("   \u2705 Linear Regression")
        print("   \u2705 Ridge Regression")
        print("   \u2705 Lasso Regression")
        print("   \u2705 ElasticNet")
        print("   \u2705 Bayesian Ridge")
        print("   \u2705 Decision Tree")
        print("   \u2705 Random Forest")
        print("   \u2705 KNN Regressor")
        
        print("\n\u1F4A1 NEXT STEPS:")
        print("   1. Open Sklearn_Regression_Models_Training.ipynb")
        print("   2. Load saved models and create visualizations")
        
    except FileNotFoundError:
        print(f"\n\u274C File not found: {file_path}")
        print("\nPlease update the file path in the script")
        
    except Exception as e:
        print(f"\n\u274C An error occurred: {e}")