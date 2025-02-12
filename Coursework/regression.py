import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load and prepare data
def load_data(file_path):
    """Load data and return both original and processed versions"""
    # Load original data for EDA
    original_data = pd.read_csv(file_path)
    
    # Create processed data with dummy variables for modeling
    processed_data = pd.read_csv(file_path)
    categorical_columns = ['sex', 'smoker', 'region']
    processed_data = pd.get_dummies(processed_data, columns=categorical_columns)
    
    return original_data, processed_data

# 2. Split features and target
def prepare_data(data):
    X = data.drop('charges', axis=1)
    y = data['charges']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# 4. Train and evaluate models
def train_linear_regression(X_train, X_test, y_train, y_test):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return lr_model, r2, rmse

def train_ridge_regression(X_train, X_test, y_train, y_test):
    # Define parameter grid
    param_grid = {
        'alpha': [0.1, 1.0, 10.0],
        'solver': ['svd', 'cholesky', 'lsqr']
    }
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return best_model, r2, rmse, grid_search.best_params_, grid_search.best_score_

# 5. Feature selection
def select_features(X_train, X_test, y_train, feature_names):
    # RFE for Linear Regression
    rfe_lr = RFE(estimator=LinearRegression(), n_features_to_select=5)
    X_train_lr_selected = rfe_lr.fit_transform(X_train, y_train)
    X_test_lr_selected = rfe_lr.transform(X_test)
    selected_features_lr = feature_names[rfe_lr.support_].tolist()
    
    # SelectKBest for Ridge
    selector_ridge = SelectKBest(score_func=f_regression, k=5)
    X_train_ridge_selected = selector_ridge.fit_transform(X_train, y_train)
    X_test_ridge_selected = selector_ridge.transform(X_test)
    selected_features_ridge = feature_names[selector_ridge.get_support()].tolist()
    
    return (X_train_lr_selected, X_test_lr_selected, selected_features_lr,
            X_train_ridge_selected, X_test_ridge_selected, selected_features_ridge)

def perform_eda(data):
    # Set a basic style that's guaranteed to work
    plt.style.use('default')
    
    # Create a figure with subplots
    plt.figure(figsize=(20, 15))
    
    # 1. Distribution of Insurance Charges
    plt.subplot(3, 2, 1)
    sns.histplot(data['charges'], kde=True)
    plt.title('Distribution of Insurance Charges')
    plt.xlabel('Charges')
    plt.ylabel('Count')
    
    # 2. Age vs Charges
    plt.subplot(3, 2, 2)
    sns.scatterplot(data=data, x='age', y='charges', hue='smoker')
    plt.title('Age vs Charges (Colored by Smoking Status)')
    
    # 3. BMI vs Charges
    plt.subplot(3, 2, 3)
    sns.scatterplot(data=data, x='bmi', y='charges', hue='smoker')
    plt.title('BMI vs Charges (Colored by Smoking Status)')
    
    # 4. Box Plot of Charges by Region
    plt.subplot(3, 2, 4)
    sns.boxplot(data=data, x='region', y='charges')
    plt.title('Charges Distribution by Region')
    
    # 5. Box Plot of Charges by Sex
    plt.subplot(3, 2, 5)
    sns.boxplot(data=data, x='sex', y='charges')
    plt.title('Charges Distribution by Sex')
    
    # 6. Correlation Heatmap
    plt.subplot(3, 2, 6)
    numeric_cols = ['age', 'bmi', 'children', 'charges']
    correlation = data[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.show()
    
    # Additional plots on separate figures
    
    # 7. Charges by Smoking Status
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='smoker', y='charges')
    plt.title('Charges Distribution by Smoking Status')
    plt.show()
    
    # 8. Charges by Number of Children
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='children', y='charges')
    plt.title('Charges Distribution by Number of Children')
    plt.show()
    
    # 9. Pair Plot for Numerical Variables
    sns.pairplot(data[numeric_cols])
    plt.suptitle('Pair Plot of Numerical Variables', y=1.02)
    plt.show()

def main():
    # Load both versions of the data
    original_data, processed_data = load_data('/content/insurance.csv')
    
    # Perform EDA on original data
    perform_eda(original_data)
    
    # Continue with modeling using processed data
    X_train, X_test, y_train, y_test = prepare_data(processed_data)
    
    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Feature selection
    (X_train_lr, X_test_lr, selected_features_lr,
     X_train_ridge, X_test_ridge, selected_features_ridge) = select_features(
        X_train_scaled, X_test_scaled, y_train, X_train.columns)
    
    # Train and evaluate Linear Regression
    lr_model, lr_r2, lr_rmse = train_linear_regression(
        X_train_lr, X_test_lr, y_train, y_test)
    
    # Train and evaluate Ridge Regression
    ridge_model, ridge_r2, ridge_rmse, best_params, best_cv_score = train_ridge_regression(
        X_train_ridge, X_test_ridge, y_train, y_test)
    
    # Calculate CV scores for Linear Regression
    cv_scores = cross_val_score(LinearRegression(), X_train_lr, y_train, cv=5)
    
    # Print results
    print("\nCustom Model Performance:")
    print(f"R2 Score: {lr_r2}")
    print(f"RMSE: {lr_rmse}")
    
    print("\nModel Performances:")
    print("Linear Regression:")
    print(f"R2 Score: {lr_r2}")
    print(f"RMSE: {lr_rmse}")
    
    print("\nRidge Regression:")
    print(f"R2 Score: {ridge_r2}")
    print(f"RMSE: {ridge_rmse}")
    
    print(f"\nLinear Regression CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean()}")
    
    print(f"\nBest Ridge Parameters: {best_params}")
    print(f"Best CV Score: {best_cv_score}")
    
    print(f"\nSelected Features (Linear Regression): {selected_features_lr}")
    print(f"Selected Features (Ridge): {selected_features_ridge}")

if __name__ == "__main__":
    main()
