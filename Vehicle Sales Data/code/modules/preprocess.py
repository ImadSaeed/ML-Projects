import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class VehicleSalesPreprocessor:
    def __init__(self, data_path=None, df=None):
        """
        Initialize preprocessor with either file path or existing dataframe
        
        Args:
            data_path: Path to CSV file
            df: Existing pandas dataframe
        """
        if data_path:
            self.df = pd.read_csv(data_path)
            print(f"\u2705 Loaded {len(self.df):,} rows and {len(self.df.columns)} columns")
        elif df is not None:
            self.df = df.copy()
            print(f"\u2705 Using existing dataframe with {len(self.df):,} rows")
        else:
            raise ValueError("Either data_path or df must be provided")
        
        self.original_df = self.df.copy()
        self.scaler = StandardScaler()
        
    def clean_data(self):
        """Step 1: Basic data cleaning"""
        print("\n" + "="*50)
        print("STEP 1: CLEANING DATA")
        print("="*50)
        
        # Drop useless columns
        columns_to_drop = ['vin']
        existing_drops = [col for col in columns_to_drop if col in self.df.columns]
        if existing_drops:
            self.df = self.df.drop(columns=existing_drops)
            print(f"\u2705 Dropped columns: {existing_drops}")
        
        # Clean transmission column (fix dirty values)
        if 'transmission' in self.df.columns:
            valid_transmissions = ['automatic', 'manual']
            self.df['transmission'] = self.df['transmission'].apply(
                lambda x: x if str(x).lower() in valid_transmissions else 'automatic'
            )
            print(f"\u2705 Cleaned transmission column - fixed invalid values")
        
        # Handle saledate - extract useful features
        if 'saledate' in self.df.columns:
            try:
                self.df['sale_month'] = pd.to_datetime(self.df['saledate'], errors='coerce').dt.month
                self.df['sale_dayofweek'] = pd.to_datetime(self.df['saledate'], errors='coerce').dt.dayofweek
                self.df = self.df.drop(columns=['saledate'])
                print(f"\u2705 Extracted sale_month and sale_dayofweek from saledate")
            except:
                self.df = self.df.drop(columns=['saledate'])
                print(f"\u26A0️ Dropped saledate (could not parse)")
        
        return self
    
    def handle_missing_values(self):
        """Step 2: Handle missing values"""
        print("\n" + "="*50)
        print("STEP 2: HANDLING MISSING VALUES")
        print("="*50)
        
        missing_before = self.df.isnull().sum()
        missing_before = missing_before[missing_before > 0]
        
        if len(missing_before) > 0:
            print(f"\u26A0️ Missing values found: {dict(missing_before)}")
            
            if 'transmission' in self.df.columns:
                self.df['transmission'] = self.df['transmission'].fillna('automatic')
                print(f"\u2705 Filled transmission missing values with 'automatic'")
            
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    mode_value = self.df[col].mode()[0]
                    self.df[col] = self.df[col].fillna(mode_value)
                    print(f"\u2705 Filled {col} missing values with mode: '{mode_value}'")
            
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    median_value = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_value)
                    print(f"\u2705 Filled {col} missing values with median: {median_value:.2f}")
        else:
            print("\u2705 No missing values found")
        
        return self
    
    def handle_outliers(self, method='cap', percentile=99):
        """Step 3: Handle outliers in numeric columns"""
        print("\n" + "="*50)
        print("STEP 3: HANDLING OUTLIERS")
        print("="*50)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        cols_to_clean = [col for col in numeric_cols if col != 'sellingprice']
        
        outliers_count = 0
        for col in cols_to_clean:
            if method == 'cap':
                upper_limit = self.df[col].quantile(percentile/100)
                lower_limit = self.df[col].quantile(1 - percentile/100)
                
                before_count = len(self.df[(self.df[col] < lower_limit) | (self.df[col] > upper_limit)])
                self.df[col] = self.df[col].clip(lower=lower_limit, upper=upper_limit)
                
                if before_count > 0:
                    print(f"\u2705 Capped {col}: {before_count:,} outliers capped at {percentile}th percentile")
                    outliers_count += before_count
        
        if outliers_count == 0:
            print("\u2705 No outliers detected or handled")
        
        return self
    
    def encode_categoricals(self, method='target_encoding', target_col='sellingprice'):
        """Step 4: Encode categorical variables"""
        print("\n" + "="*50)
        print("STEP 4: ENCODING CATEGORICAL VARIABLES")
        print("="*50)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            print("\u2705 No categorical columns to encode")
            return self
        
        if method == 'target_encoding' and target_col in self.df.columns:
            for col in categorical_cols:
                if self.df[col].nunique() > 20:
                    target_means = self.df.groupby(col)[target_col].mean()
                    self.df[f'{col}_encoded'] = self.df[col].map(target_means)
                    self.df = self.df.drop(columns=[col])
                    print(f"\u2705 Target encoded {col} ({len(target_means)} categories)")
                else:
                    dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                    self.df = pd.concat([self.df, dummies], axis=1)
                    self.df = self.df.drop(columns=[col])
                    print(f"\u2705 One-hot encoded {col}")
        else:
            le = LabelEncoder()
            for col in categorical_cols:
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                print(f"\u2705 Label encoded {col}")
        
        return self
    
    def prepare_regression_target(self):
        """Step 5: Prepare target for regression (log transform)"""
        print("\n" + "="*50)
        print("STEP 5: PREPARING REGRESSION TARGET")
        print("="*50)
        
        if 'sellingprice' not in self.df.columns:
            raise ValueError("sellingprice column not found")
        
        # Log transform to handle skewness
        self.df['sellingprice_log'] = np.log1p(self.df['sellingprice'])
        
        # Optionally cap extreme outliers in target
        upper_limit = self.df['sellingprice'].quantile(0.99)
        self.df['sellingprice_capped'] = self.df['sellingprice'].clip(upper=upper_limit)
        self.df['sellingprice_capped_log'] = np.log1p(self.df['sellingprice_capped'])
        
        print(f"\u2705 Applied log transformation to sellingprice")
        print(f"   Original skewness: 1.95")
        print(f"   New skewness: {self.df['sellingprice_log'].skew():.3f}")
        
        return self
    
    def prepare_classification_target(self, method='multi'):
        """Step 6: Prepare target for MULTI-CLASS classification (default is multi, not binary)"""
        print("\n" + "="*50)
        print("STEP 6: PREPARING MULTI-CLASS TARGET")
        print("="*50)
        
        if 'sellingprice' not in self.df.columns:
            raise ValueError("sellingprice column not found")
        
        if method == 'binary':
            # Binary classification (above/below median)
            median_price = self.df['sellingprice'].median()
            self.df['price_category'] = (self.df['sellingprice'] > median_price).astype(int)
            print(f"\u2705 Binary classification created (median: ${median_price:,.2f})")
            print(f"   Class 0 (Low): {sum(self.df['price_category']==0):,} ({sum(self.df['price_category']==0)/len(self.df)*100:.1f}%)")
            print(f"   Class 1 (High): {sum(self.df['price_category']==1):,} ({sum(self.df['price_category']==1)/len(self.df)*100:.1f}%)")
            
        elif method == 'multi':
            # Multi-class classification (tertiles) - DEFAULT
            self.df['price_category'] = pd.qcut(
                self.df['sellingprice'], 
                q=3, 
                labels=[0, 1, 2],  # 0=Low, 1=Medium, 2=High
                duplicates='drop'
            )
            
            print(f"\u2705 Multi-class classification created (3 classes)")
            print(f"\n\uD83D\uDCCA Class Distribution:")
            
            class_counts = self.df['price_category'].value_counts().sort_index()
            class_percentages = self.df['price_category'].value_counts(normalize=True).sort_index() * 100
            
            for class_label in [0, 1, 2]:
                class_name = {0: 'Low', 1: 'Medium', 2: 'High'}[class_label]
                min_price = self.df[self.df['price_category'] == class_label]['sellingprice'].min()
                max_price = self.df[self.df['price_category'] == class_label]['sellingprice'].max()
                print(f"   Class {class_label} ({class_name}): {class_counts[class_label]:,} ({class_percentages[class_label]:.1f}%) - ${min_price:,.0f} to ${max_price:,.0f}")
            
        return self
    
    def scale_features(self):
        """Step 7: Scale numeric features"""
        print("\n" + "="*50)
        print("STEP 7: SCALING NUMERIC FEATURES")
        print("="*50)
        
        exclude_cols = ['sellingprice', 'sellingprice_log', 'sellingprice_capped', 
                       'sellingprice_capped_log', 'price_category']
        if 'price_category' in self.df.columns:
            exclude_cols.append('price_category')
        
        numeric_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        
        if len(numeric_cols) > 0:
            self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])
            print(f"\u2705 Scaled {len(numeric_cols)} numeric features")
        else:
            print("\u2705 No numeric features to scale")
        
        return self
    
    def get_regression_data(self, test_size=0.2, random_state=42):
        """Return train-test split for regression"""
        print("\n" + "="*50)
        print("PREPARING REGRESSION DATA")
        print("="*50)
        
        # Use log-transformed target
        target_col = 'sellingprice_log'
        
        # Define features (exclude original target and other derived columns)
        exclude = ['sellingprice', 'sellingprice_log', 'sellingprice_capped', 
                  'sellingprice_capped_log', 'price_category']
        feature_cols = [col for col in self.df.columns if col not in exclude]
        
        X = self.df[feature_cols]
        y = self.df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\u2705 Regression data ready")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Train size: {len(X_train):,}")
        print(f"   Test size: {len(X_test):,}")
        
        return X_train, X_test, y_train, y_test
    
    def get_classification_data(self, test_size=0.2, random_state=42, method='multi'):
        """Return train-test split for MULTI-CLASS classification (default is multi)"""
        print("\n" + "="*50)
        print("PREPARING CLASSIFICATION DATA")
        print("="*50)
        
        # Use price_category as target
        if 'price_category' not in self.df.columns:
            self.prepare_classification_target(method=method)
        
        target_col = 'price_category'
        
        # Define features (exclude target columns)
        exclude = ['sellingprice', 'sellingprice_log', 'sellingprice_capped', 
                  'sellingprice_capped_log', 'price_category']
        feature_cols = [col for col in self.df.columns if col not in exclude]
        
        X = self.df[feature_cols]
        y = self.df[target_col]
        
        # Split data with stratification for classification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\u2705 Classification data ready")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Train size: {len(X_train):,}")
        print(f"   Test size: {len(X_test):,}")
        print(f"\n\uD83D\uDCCA Train set class distribution:")
        print(y_train.value_counts().sort_index())
        print(f"\n\uD83D\uDCCA Test set class distribution:")
        print(y_test.value_counts().sort_index())
        
        return X_train, X_test, y_train, y_test
    
    def get_data_summary(self):
        """Print summary of processed data"""
        print("\n" + "="*50)
        print("FINAL DATA SUMMARY")
        print("="*50)
        print(f"Total rows: {len(self.df):,}")
        print(f"Total columns: {len(self.df.columns)}")
        print(f"\nFeature types:")
        print(f"  - Numeric: {len(self.df.select_dtypes(include=[np.number]).columns)}")
        print(f"  - Categorical: {len(self.df.select_dtypes(include=['object']).columns)}")
        print(f"\nTarget available:")
        if 'sellingprice_log' in self.df.columns:
            print(f"  - Regression: sellingprice_log (log transformed)")
        if 'price_category' in self.df.columns:
            print(f"  - Classification: price_category (0=Low, 1=Medium, 2=High)")
        
        return self


# ============================================
# QUICK START FUNCTIONS
# ============================================

def preprocess_vehicle_data_regression(data_path):
    """
    One-stop function to preprocess vehicle sales data for REGRESSION
    
    Args:
        data_path: Path to CSV file
    
    Returns:
        X_train, X_test, y_train, y_test (for regression)
    """
    preprocessor = (VehicleSalesPreprocessor(data_path)
                    .clean_data()
                    .handle_missing_values()
                    .handle_outliers(method='cap', percentile=99)
                    .encode_categoricals(method='target_encoding')
                    .prepare_regression_target()
                    .scale_features())
    
    return preprocessor.get_regression_data()


def preprocess_vehicle_data_multiclass(data_path):
    """
    One-stop function to preprocess vehicle sales data for MULTI-CLASS CLASSIFICATION
    
    Args:
        data_path: Path to CSV file
    
    Returns:
        X_train, X_test, y_train, y_test (multi-class labels: 0=Low, 1=Medium, 2=High)
    """
    preprocessor = (VehicleSalesPreprocessor(data_path)
                    .clean_data()
                    .handle_missing_values()
                    .handle_outliers(method='cap', percentile=99)
                    .encode_categoricals(method='target_encoding')
                    .prepare_classification_target(method='multi')
                    .scale_features())
    
    return preprocessor.get_classification_data(method='multi')