# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 1. Load and explore the data
df = pd.read_csv('insurance.csv')

# Basic EDA
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Enhanced EDA for Regression Task
print("\n=== Enhanced EDA for Insurance Dataset ===")
# Distribution plots
plt.figure(figsize=(15, 10))
for i, col in enumerate(df.select_dtypes(include=['float64', 'int64']).columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Box plots for categorical variables
plt.figure(figsize=(15, 5))
categorical_cols = ['sex', 'smoker', 'region']
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=col, y='charges', data=df)
    plt.title(f'{col} vs charges')
plt.tight_layout()
plt.show()

# 2. Data Preprocessing
# Convert categorical variables to numeric
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])

# Visualizations
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Split features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Build Linear Regression from Scratch
class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Gradient descent
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train custom model
custom_model = CustomLinearRegression()
custom_model.fit(X_train_scaled, y_train)
custom_pred = custom_model.predict(X_test_scaled)
print("\nCustom Model Performance:")
print(f"R2 Score: {r2_score(y_test, custom_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, custom_pred))}")

# 4. Build Primary Models
# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# Model 2: Ridge Regression
ridge_model = Ridge()
ridge_model.fit(X_train_scaled, y_train)
ridge_pred = ridge_model.predict(X_test_scaled)

print("\nModel Performances:")
print("Linear Regression:")
print(f"R2 Score: {r2_score(y_test, lr_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred))}")
print("\nRidge Regression:")
print(f"R2 Score: {r2_score(y_test, ridge_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, ridge_pred))}")

# Enhanced Hyperparameter Optimization
# Model 1: Linear Regression with cross-validation
lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5)
print("\nLinear Regression CV Scores:", lr_cv_scores)
print("Mean CV Score:", lr_cv_scores.mean())

# Model 2: Ridge Regression with expanded parameter grid
ridge_params = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'solver': ['auto', 'svd', 'cholesky']
}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='r2')
ridge_grid.fit(X_train_scaled, y_train)
print("\nBest Ridge Parameters:", ridge_grid.best_params_)
print("Best CV Score:", ridge_grid.best_score_)

# Feature Selection for both models
# Using different methods for comparison

# RFE for Linear Regression
rfe_lr = RFE(estimator=LinearRegression(), n_features_to_select=5)
X_train_lr_selected = rfe_lr.fit_transform(X_train_scaled, y_train)
X_test_lr_selected = rfe_lr.transform(X_test_scaled)
selected_features_lr = X_train.columns[rfe_lr.support_].tolist()
print("\nSelected Features (Linear Regression):", selected_features_lr)

# SelectKBest for Ridge
selector_ridge = SelectKBest(score_func=f_regression, k=5)
X_train_selected = selector_ridge.fit_transform(X_train_scaled, y_train)
X_test_selected = selector_ridge.transform(X_test_scaled)
selected_features = X_train.columns[selector_ridge.get_support()].tolist()
print("Selected Features (Ridge):", selected_features)

# 7. Final Model with Selected Features
final_ridge = Ridge(**ridge_grid.best_params_)
final_ridge.fit(X_train_selected, y_train)
final_pred = final_ridge.predict(X_test_selected)

print("\nFinal Model Performance:")
print(f"R2 Score: {r2_score(y_test, final_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, final_pred))}")

# Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, final_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Insurance Charges')
plt.show()

# Regression Conclusion Section
print("\n=== Regression Analysis Conclusions ===")
models_comparison = {
    'Custom Linear Regression': {'R2': r2_score(y_test, custom_pred), 
                               'RMSE': np.sqrt(mean_squared_error(y_test, custom_pred))},
    'Linear Regression': {'R2': r2_score(y_test, lr_pred), 
                         'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred))},
    'Ridge Regression': {'R2': r2_score(y_test, ridge_pred), 
                        'RMSE': np.sqrt(mean_squared_error(y_test, ridge_pred))},
    'Final Ridge (Selected Features)': {'R2': r2_score(y_test, final_pred), 
                                      'RMSE': np.sqrt(mean_squared_error(y_test, final_pred))}
}

# Create comparison DataFrame
comparison_df = pd.DataFrame(models_comparison).T
print("\nModel Comparison:")
print(comparison_df)

# Feature importance analysis for final model
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': final_ridge.coef_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Coefficient', key=abs, ascending=False))

# Classification Task
print("\n=== Classification Task ===")

# Load breast cancer dataset
cancer = load_breast_cancer()
X_clf = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_clf = cancer.target

# Split the data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Scale features
X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

# Feature selection for classification
selector_clf = SelectKBest(k=10)
X_train_clf_selected = selector_clf.fit_transform(X_train_clf_scaled, y_train_clf)
X_test_clf_selected = selector_clf.transform(X_test_clf_scaled)

# Get selected feature names
selected_features_clf = X_clf.columns[selector_clf.get_support()].tolist()
print("\nSelected Classification Features:", selected_features_clf)

# Train multiple classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

# Hyperparameter grids
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20]
    },
    'SVM': {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'linear']
    }
}

# Train and evaluate classifiers
clf_results = {}
for name, clf in classifiers.items():
    # Perform GridSearchCV
    grid_search = GridSearchCV(clf, param_grids[name], cv=5)
    grid_search.fit(X_train_clf_selected, y_train_clf)
    
    # Get best model
    best_model = grid_search.best_estimator_
    y_pred_clf = best_model.predict(X_test_clf_selected)
    
    # Store results
    clf_results[name] = {
        'Best Parameters': grid_search.best_params_,
        'Accuracy': accuracy_score(y_test_clf, y_pred_clf),
        'Model': best_model
    }

# Print classification results
print("\n=== Classification Results ===")
for name, results in clf_results.items():
    print(f"\n{name}:")
    print(f"Best Parameters: {results['Best Parameters']}")
    print(f"Accuracy: {results['Accuracy']:.4f}")
    print("\nClassification Report:")
    y_pred = results['Model'].predict(X_test_clf_selected)
    print(classification_report(y_test_clf, y_pred))

# Visualize confusion matrix for best model
best_clf_name = max(clf_results.items(), key=lambda x: x[1]['Accuracy'])[0]
best_clf = clf_results[best_clf_name]['Model']
y_pred_best = best_clf.predict(X_test_clf_selected)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test_clf, y_pred_best), 
            annot=True, 
            fmt='d',
            cmap='Blues')
plt.title(f'Confusion Matrix - {best_clf_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Final conclusions
print("\n=== Final Conclusions ===")
print("\nRegression Task (Insurance Charges):")
print("- Best performing model:", comparison_df['R2'].idxmax())
print("- Most important features:", ", ".join(feature_importance['Feature'].head(3).tolist()))
print(f"- Best R2 Score: {comparison_df['R2'].max():.4f}")

print("\nClassification Task (Breast Cancer):")
print("- Best performing model:", best_clf_name)
print("- Most important features:", ", ".join(selected_features_clf[:3]))
print(f"- Best Accuracy: {clf_results[best_clf_name]['Accuracy']:.4f}")

# Enhanced EDA for Classification Task
print("\n=== Enhanced EDA for Breast Cancer Dataset ===")
X_clf_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
X_clf_df['target'] = cancer.target

# Correlation matrix for top features
plt.figure(figsize=(12, 8))
top_features = X_clf_df.corr()['target'].sort_values(ascending=False)[:10].index
sns.heatmap(X_clf_df[top_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Top Features')
plt.tight_layout()
plt.show()

# Build Logistic Regression from Scratch
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Gradient descent
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)

# Train custom logistic regression
custom_log_reg = CustomLogisticRegression()
custom_log_reg.fit(X_train_clf_scaled, y_train_clf)
custom_pred_clf = custom_log_reg.predict(X_test_clf_scaled)
print("\nCustom Logistic Regression Accuracy:", 
      accuracy_score(y_test_clf, custom_pred_clf))
      