# 🚗 Vehicle Sales Data Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data)
![Records](https://img.shields.io/badge/Records-558%2C837-f97316?style=flat)
![Models](https://img.shields.io/badge/Models-22%20Total-8b5cf6?style=flat)
![Best R²](https://img.shields.io/badge/Best%20R%C2%B2-95.44%25-brightgreen?style=flat)

> A complete end-to-end machine learning pipeline for predicting used car prices from US auto auction data (2014–2015), covering EDA, preprocessing, regression, and classification across **22 models**.

> Full Project:- [![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/imadsaeed123/vehicle-sale-data)

## 📑 Table of Contents

- [Overview](#-overview)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Key Insights](#-key-insights)
- [Technologies](#-technologies)
- [Dataset](#-dataset)
- [Usage](#-usage)
- [License](#-license)

---

## 🔍 Overview

This project tackles used car price prediction as both a **regression problem** (exact price) and a **classification problem** (price tier: Low / Medium / High).

The full pipeline includes:

- **Exploratory Data Analysis (EDA)** — distributions, correlations, outliers
- **Data Preprocessing** — cleaning, encoding, scaling, log transformation
- **12 Regression Models** — from Linear Regression to HistGradientBoosting
- **10 Classification Models** — from Logistic Regression to LightGBM
- **Feature Importance Analysis** — understanding what drives price
- **Model Comparison** — leaderboards with R², MAPE, Accuracy, F1, and training time

---

## 📊 Results

### 🏆 Best Models

| Task               | Model                | Score                 |
| ------------------ | -------------------- | --------------------- |
| **Regression**     | HistGradientBoosting | R² = **95.44%**       |
| **Classification** | LightGBM             | Accuracy = **93.00%** |

---

### 📈 Regression Leaderboard

| Rank | Model                | R² Score   | MAPE   | Time (s) |
| ---- | -------------------- | ---------- | ------ | -------- |
| 🥇   | HistGradientBoosting | **0.9544** | 12.11% | 9.56     |
| 🥈   | LightGBM             | 0.9529     | 12.31% | 4.12     |
| 🥉   | XGBoost              | 0.9528     | 12.32% | 4.86     |
| 4    | Random Forest        | 0.9500     | 12.51% | 230.12   |
| 5    | Decision Tree        | 0.9155     | 15.41% | 6.64     |
| 6    | KNN                  | 0.9110     | 18.39% | 0.14     |
| 7    | Linear / Ridge       | 0.8489     | 28.33% | < 1      |
| 8    | ElasticNet           | 0.3706     | 90.12% | 0.59     |

---

### 📊 Classification Leaderboard

| Rank | Model               | Accuracy   | F1-Score   | Time (s) |
| ---- | ------------------- | ---------- | ---------- | -------- |
| 🥇   | LightGBM            | **93.00%** | **93.01%** | 14.64    |
| 🥈   | XGBoost             | 92.91%     | 92.92%     | 18.31    |
| 🥉   | Random Forest       | 92.80%     | 92.82%     | 50.05    |
| 4    | Decision Tree       | 90.72%     | 90.74%     | 6.86     |
| 5    | Logistic Regression | 89.93%     | 89.88%     | 3.84     |
| 6    | KNN                 | 89.12%     | 89.16%     | 0.12     |

---

## 📁 Project Structure

```
Vehicle-Sales-Data/
│
├── code/
│   ├── modules/
│   │   ├── __init__.py
│   │   └── preprocess.py                              # Shared preprocessing pipeline
│   │
│   ├── Script/
│   │   ├── Sklearn_Regression_models.py               # 8 sklearn regression models
│   │   ├── Regression_Gradient_Boosting_Algo.py       # XGBoost, LightGBM, CatBoost (reg)
│   │   ├── Sklearn_Classification_models.py           # 6 sklearn classification models
│   │   └── Classification_Gradient_Boosting_Algo.py  # Gradient boosting (clf)
│   │
│   └── Notebooks/
│       ├── EDA.ipynb
│       ├── sklearn-regression-models-training.ipynb
│       ├── reg-gradient-boosting-training.ipynb
│       ├── sklearn-classification-models-training.ipynb
│       └── classification-gradient-boosting-training.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/vehicle-sales-analysis.git
cd vehicle-sales-analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download `car_prices.csv` from [Kaggle](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data) and place it in the project root.

### 4. Run EDA

```bash
jupyter notebook code/Notebooks/EDA.ipynb
```

### 5. Train Models

```bash
# Regression
python code/Script/Sklearn_Regression_models.py
python code/Script/Regression_Gradient_Boosting_Algo.py

# Classification
python code/Script/Sklearn_Classification_models.py
python code/Script/Classification_Gradient_Boosting_Algo.py
```

---

## 🔧 Preprocessing Pipeline

The `preprocess.py` module provides two ready-to-use functions that handle the full preprocessing flow:

| Step                 | Details                                                                    |
| -------------------- | -------------------------------------------------------------------------- |
| **Data Cleaning**    | Drop VIN column, fix transmission inconsistencies, extract date features   |
| **Missing Values**   | Mode imputation for categorical, median for numeric columns                |
| **Outlier Handling** | Values capped at the 99th percentile                                       |
| **Encoding**         | Target encoding for high-cardinality features, One-Hot for low-cardinality |
| **Target Transform** | Log transform on selling price for regression (reduces skew)               |
| **Feature Scaling**  | StandardScaler applied to all numeric features                             |
| **Train/Test Split** | 80/20 split; stratified sampling for classification                        |

---

## 💡 Key Insights

### Most Important Feature

**MMR (Manheim Market Report)** dominates all models with **90–95% feature importance**, acting as a near-ground-truth wholesale price estimate.

```
Feature Importance — Random Forest (Regression)
─────────────────────────────────────────────────
  mmr              ████████████████████  94.59%
  condition        ██                     1.61%
  seller_encoded   ██                     0.83%
  odometer         █                      0.59%
  (others)                                2.38%
─────────────────────────────────────────────────
```

### Price Distribution

The raw selling price is heavily right-skewed. A **log transformation** normalizes the distribution, improving regression model performance across the board.

```
Before (skewed)              After log transform (approx. normal)
  |█                                |   ██
  |███                              |  ████
  |█████                            | ██████
  |████████                         |████████
  └──────────────                   └──────────────
  $1          $230k               log(1)    log(230k)
```

---

## 🛠️ Technologies

| Category            | Libraries                                                                                                 |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| **Data Processing** | Pandas, NumPy                                                                                             |
| **Visualization**   | Matplotlib, Seaborn                                                                                       |
| **ML — Sklearn**    | LinearRegression, Ridge, Lasso, KNN, DecisionTree, RandomForest, HistGradientBoosting, LogisticRegression |
| **ML — Boosting**   | XGBoost, LightGBM, CatBoost                                                                               |
| **Environment**     | Jupyter Notebook, Kaggle Kernels                                                                          |

---

## 📋 Dataset

| Property        | Details                                                                                          |
| --------------- | ------------------------------------------------------------------------------------------------ |
| **Source**      | [Kaggle — Syed Anwar Afridi](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data) |
| **Records**     | 558,837 rows                                                                                     |
| **Features**    | 16 columns                                                                                       |
| **Period**      | US Auto Auctions, 2014–2015                                                                      |
| **Price Range** | $1 – $230,000 (Median: $12,100)                                                                  |

### Feature Descriptions

| Feature        | Description                                    |
| -------------- | ---------------------------------------------- |
| `year`         | Manufacturing year of the vehicle              |
| `make`         | Car manufacturer (e.g., Ford, Toyota)          |
| `model`        | Specific vehicle model                         |
| `condition`    | Auction condition score (1–49)                 |
| `odometer`     | Mileage at time of sale                        |
| `mmr`          | Manheim Market Report wholesale price estimate |
| `sellingprice` | **Target variable** — final auction sale price |

---

## ⚙️ Usage

### Regression — Predict Exact Price

```python
from modules.preprocess import preprocess_vehicle_data_regression

X_train, X_test, y_train, y_test = preprocess_vehicle_data_regression('car_prices.csv')

# Note: y values are log-transformed — use np.expm1() to convert predictions back to dollars
```

### Classification — Predict Price Tier

```python
from modules.preprocess import preprocess_vehicle_data_multiclass

X_train, X_test, y_train, y_test = preprocess_vehicle_data_multiclass('car_prices.csv')

# Labels:  0 = Low  |  1 = Medium  |  2 = High
```

---

## 📓 Notebooks Guide

| Notebook                                          | Description                                                             |
| ------------------------------------------------- | ----------------------------------------------------------------------- |
| `EDA.ipynb`                                       | Full exploratory analysis — distributions, correlations, missing values |
| `sklearn-regression-models-training.ipynb`        | Train and compare 8 sklearn regression models                           |
| `reg-gradient-boosting-training.ipynb`            | XGBoost, LightGBM, CatBoost for regression                              |
| `sklearn-classification-models-training.ipynb`    | Train and compare 6 sklearn classifiers                                 |
| `classification-gradient-boosting-training.ipynb` | Gradient boosting classifiers with full evaluation                      |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- **Dataset**: [Syed Anwar Afridi](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data) on Kaggle
- **MMR Benchmark**: Manheim Market Report wholesale pricing
- **Community**: Kaggle notebooks and the open-source ML community

---

## 📬 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Kaggle**: [@yourkaggleusername](https://www.kaggle.com/yourkaggleusername)

---

<p align="center">If this project helped you, please consider giving it a ⭐ on GitHub!</p>
