""""
Training Gradient Boosting regression models
Models: HistGradientBoosting, XGBoost, LightGBM, CatBoost
SAVES: Models (.pkl), Metrics (.csv), Training details (.json)
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

# Import preprocessing function
from modules.preprocess import preprocess_vehicle_data_regression

# Import gradient boosting libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


class GradientBoostingModels:
    """
    Class to train and evaluate Gradient Boosting regression models
    """
    
    def __init__(self, data_path):
        """
        Initialize with data path and load preprocessed data
        """
        print("="*60)
        print("GRADIENT BOOSTING REGRESSION MODELS")
        print("="*60)
        
        # Load preprocessed data
        print("\n\u1F4C2 Loading preprocessed data...")
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_vehicle_data_regression(data_path)
        
        print(f"\n\u2705 Data loaded successfully!")
        print(f"   Training set size: {self.X_train.shape}")
        print(f"   Test set size: {self.X_test.shape}")
        
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
        
        # Calculate MAPE
        y_test_actual = self._inverse_transform_price(self.y_test)
        y_pred_actual = self._inverse_transform_price(y_pred_test)
        mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
        
        metrics = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': mape,
            'train_time': train_time
        }
        
        # Print metrics
        print(f"\n\u1F4CA Results:")
        print(f"   Training R\u00B2: {train_r2:.4f}")
        print(f"   Test R\u00B2: {test_r2:.4f}")
        print(f"   Test MAE: ${self._inverse_transform_price(test_mae):,.2f}")
        print(f"   Test RMSE: ${self._inverse_transform_price(test_rmse):,.2f}")
        print(f"   Test MAPE: {mape:.2f}%")
        print(f"   Training time: {train_time:.2f} seconds")
        
        return metrics
    
    def _inverse_transform_price(self, log_value):
        """Convert log-transformed price back to actual price"""
        return np.expm1(log_value)
    
    def _save_model(self, model, filename):
        """Save trained model"""
        filepath = os.path.join(self.models_dir, filename)
        joblib.dump(model, filepath)
        print(f"   \u1F4BE Model saved: {filename}")
        
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
    
    def _save_predictions(self, model_name, y_pred):
        """Save predictions"""
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
    
    def _store_results(self, model_name, metrics):
        """Store results for comparison"""
        self.results.append({
            'Model': model_name,
            'Train R\u00B2': metrics['train_r2'],
            'Test R\u00B2': metrics['test_r2'],
            'Test MAE (\u0024)': self._inverse_transform_price(metrics['test_mae']),
            'Test RMSE (\u0024)': self._inverse_transform_price(metrics['test_rmse']),
            'Test MAPE (\u0025)': metrics['test_mape'],
            'Training Time (s)': metrics['train_time']
        })
    
    def _get_feature_importance(self, model, model_name, feature_names):
        """Get and save feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        else:
            return
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(10)
        
        print(f"\n\u1F51D Top 10 Features ({model_name}):")
        for idx, row in importance_df.iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
        
        # Save to file
        filename = f"{model_name.lower().replace(' ', '_')}_feature_importance.csv"
        filepath = os.path.join(self.evaluation_dir, filename)
        importance_df.to_csv(filepath, index=False)
    
    def train_hist_gradient_boosting(self, max_iter=200):
        """Train Sklearn's HistGradientBoostingRegressor"""
        print("\n" + "="*50)
        print(f"\u1F4C8 Training HistGradientBoosting (iterations={max_iter})")
        print("="*50)
        
        start_time = time.time()
        
        model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            random_state=42,
            verbose=1
        )
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time)
        self._save_model(model, 'hist_gradient_boosting.pkl')
        self._save_predictions('HistGradientBoosting', y_pred_test)
        self._get_feature_importance(model, 'HistGradientBoosting', self.X_train.columns)
        self._store_results(f'HistGradientBoosting (iter={max_iter})', metrics)
        
        return model, metrics
    
    def train_xgboost(self, n_estimators=200, learning_rate=0.05):
        """Train XGBoost Regressor"""
        print("\n" + "="*50)
        print(f"\u1F4C8 Training XGBoost (n_estimators={n_estimators}, lr={learning_rate})")
        print("="*50)
        
        start_time = time.time()
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time)
        self._save_model(model, 'xgboost_regressor.pkl')
        self._save_predictions('XGBoost', y_pred_test)
        self._get_feature_importance(model, 'XGBoost', self.X_train.columns)
        self._store_results(f'XGBoost (n={n_estimators})', metrics)
        
        return model, metrics
    
    def train_lightgbm(self, n_estimators=200, learning_rate=0.05):
        """Train LightGBM Regressor"""
        print("\n" + "="*50)
        print(f"\u1F4C8 Training LightGBM (n_estimators={n_estimators}, lr={learning_rate})")
        print("="*50)
        
        start_time = time.time()
        
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time)
        self._save_model(model, 'lightgbm_regressor.pkl')
        self._save_predictions('LightGBM', y_pred_test)
        self._get_feature_importance(model, 'LightGBM', self.X_train.columns)
        self._store_results(f'LightGBM (n={n_estimators})', metrics)
        
        return model, metrics
    
    def train_catboost(self, iterations=200, learning_rate=0.05):
        """Train CatBoost Regressor"""
        print("\n" + "="*50)
        print(f"\u1F4C8 Training CatBoost (iterations={iterations}, lr={learning_rate})")
        print("="*50)
        
        start_time = time.time()
        
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=6,
            random_seed=42,
            verbose= True
        )
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        metrics = self._calculate_metrics(y_pred_train, y_pred_test, train_time)
        self._save_model(model, 'catboost_regressor.pkl')
        self._save_predictions('CatBoost', y_pred_test)
        self._get_feature_importance(model, 'CatBoost', self.X_train.columns)
        self._store_results(f'CatBoost (iter={iterations})', metrics)
        
        return model, metrics
    
    def save_all_results(self):
        """Save all results to CSV and JSON"""
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('Test R\u00B2', ascending=False)
        
        csv_path = os.path.join(self.results_dir, 'gradient_boosting_comparison.csv')
        results_df.to_csv(csv_path, index=False)
        
        json_path = os.path.join(self.results_dir, 'gradient_boosting_comparison.json')
        results_df.to_json(json_path, orient='records', indent=4)
        
        print("\n" + "="*60)
        print("\u1F4CA GRADIENT BOOSTING MODEL COMPARISON")
        print("="*60)
        print(results_df.to_string(index=False))
        
        print(f"\n\u2705 Results saved to:")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")
        
        if len(results_df) > 0:
            best_model = results_df.iloc[0]
            print(f"\n\u1F3C5 BEST GRADIENT BOOSTING MODEL: {best_model['Model']}")
            print(f"   Test R\u00B2: {best_model['Test R\u00B2']:.4f}")
            print(f"   Test MAPE: {best_model['Test MAPE (\u0025)']:.2f}%")
        
        return results_df
    
    def run_all_models(self):
        """Run all gradient boosting models"""
        print("\n" + "\uD83D|*30")
        print("RUNNING ALL GRADIENT BOOSTING MODELS")
        print("\uD83D|*30")
        print("\n\u1F4CC Models to train: 4\n")
        
        # HistGradientBoosting (built-in sklearn)
        self.train_hist_gradient_boosting(max_iter=200)
        
        # XGBoost
        self.train_xgboost(n_estimators=200, learning_rate=0.05)
        
        # LightGBM
        self.train_lightgbm(n_estimators=200, learning_rate=0.05)
        
        # CatBoost
        self.train_catboost(iterations=200, learning_rate=0.05)
        
        results_df = self.save_all_results()
        return results_df


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("\u1F3E0 GRADIENT BOOSTING REGRESSION MODELS")
    print("="*60)
    
    # Your Kaggle path
    file_path = '/kaggle/input/datasets/imadsaeed123/vehicle-sale-data/Vehicle Sales Data/archive/car_prices.csv'
    
    try:
        trainer = GradientBoostingModels(file_path)
        results = trainer.run_all_models()
        
        print("\n" + "="*60)
        print("\u2705 ALL GRADIENT BOOSTING MODELS TRAINED SUCCESSFULLY!")
        print("="*60)
        
        print("\n\u1F4C2 FILES SAVED:")
        print("   \u1F4C1 /models/")
        print("      ├── hist_gradient_boosting.pkl")
        print("      ├── xgboost_regressor.pkl")
        print("      ├── lightgbm_regressor.pkl")
        print("      └── catboost_regressor.pkl")
        
        print("\n   \u1F4C2 /results/")
        print("      └── gradient_boosting_comparison.csv")
        
        print("\n   \u1F4C1 /evaluation/")
        print("      \u1F4C1 histgradientboosting_predictions.csv")
        print("      \u1F4C2 xgboost_predictions.csv")
        print("      \u1F4C2 lightgbm_predictions.csv")
        print("      \u1F4C2 catboost_predictions.csv")
        
        print("\n\u1F4A1 NEXT STEP: Compare with Random Forest results!")
        
    except FileNotFoundError:
        print(f"\n\u274C File not found: {file_path}")
        print("Please update the file path")
        
    except Exception as e:
        print(f"\n\u274C An error occurred: {e}")