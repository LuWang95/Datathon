"""
This script loads a real estate dataset from a CSV file, preprocesses it, trains
an XGBoost regression model to predict property prices, and produces a series
of visualizations to analyze model performance and data distributions.

Steps:
1. Load data from real-estate-data.csv
2. Preprocess columns (convert size ranges, encode categories, etc.)
3. Split data into training and testing sets
4. Train and tune an XGBoost model
5. Evaluate and visualize results (scatter plots, distribution plots, etc.)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def convert_size(size_str: str) -> float:
    """
    Convert a size range string (e.g., "500-999 sqft") into a numeric average.

    Parameters
    ----------
    size_str : str
        A string representing a size range with "sqft" suffix (e.g. "500-999 sqft").

    Returns
    -------
    float
        The midpoint of the size range if the string is valid; otherwise, NaN.
    """
    if isinstance(size_str, str) and "sqft" in size_str:
        size_str = size_str.replace(" sqft", "")
        parts = size_str.split("-")
        if len(parts) == 2:
            low, high = int(parts[0]), int(parts[1])
            return (low + high) / 2
    return np.nan

# ------------------------------------------------------------------------------
# 1) LOAD DATA
# ------------------------------------------------------------------------------
"""
Load the real estate dataset from CSV and store in a pandas DataFrame.
"""

df = pd.read_csv("real-estate-data.csv")
print("Data Shape:", df.shape)

# ------------------------------------------------------------------------------
# 2) PREPROCESSING
# ------------------------------------------------------------------------------
"""
Preprocess the data by:
- Converting 'size' from range strings to numeric averages
- Filling missing values
- Encoding categorical features
- Creating engineered features like 'price_per_sqft' and 'age_bucket'
"""

# Convert 'size' if it exists in the dataset
if "size" in df.columns:
    df["size"] = df["size"].apply(convert_size)

# Fill missing values for key numeric columns
df.fillna({
    "beds": df["beds"].median(),
    "size": df["size"].median(),
    "maint": df["maint"].median(),
    "D_mkt": df["D_mkt"].median()
}, inplace=True)

# Drop rows with missing target ('price')
df.dropna(subset=["price"], inplace=True)

# Encode categorical variables
if "DEN" in df.columns:
    df["DEN"] = df["DEN"].map({"YES": 1, "No": 0})
if "parking" in df.columns:
    df["parking"] = df["parking"].map({"Yes": 1, "N": 0})

# One-hot encode 'ward' and 'exposure' if present
if "ward" in df.columns:
    df = pd.get_dummies(df, columns=["ward"], drop_first=True)
if "exposure" in df.columns:
    df = pd.get_dummies(df, columns=["exposure"], drop_first=True)

# Feature Engineering
if "size" in df.columns:
    df["price_per_sqft"] = df["price"] / df["size"]

if "building_age" in df.columns:
    df["age_bucket"] = pd.cut(
        df["building_age"],
        bins=[0, 10, 20, 30, 50, 100],
        labels=[1, 2, 3, 4, 5]
    )

if "id_" in df.columns:
    df.drop(columns=["id_"], inplace=True, errors='ignore')

# ------------------------------------------------------------------------------
# 3) MODELING
# ------------------------------------------------------------------------------
"""
Define features and target, impute remaining missing values, scale features, and
split data into training and test sets. Then train an XGBoost model with
hyperparameter tuning via GridSearchCV.
"""

# Define features (X) and target (y)
X = df.drop(columns=["price"])
y = df["price"]

# Impute any remaining missing values in X
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning for XGBoost
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}
xgb_model = xgb.XGBRegressor(random_state=42)
gs = GridSearchCV(xgb_model, param_grid, cv=3, scoring='r2', verbose=1, n_jobs=-1)
gs.fit(X_train, y_train)
best_model = gs.best_estimator_
print("Best XGBoost Parameters:", gs.best_params_)

# Evaluate model
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Optimized XGBoost R2 Score: {r2:.4f}")
print(f"MAE: {mae:,.2f}")
print(f"MSE: {mse:,.2f}")

# ------------------------------------------------------------------------------
# 4) VISUALIZATIONS
# ------------------------------------------------------------------------------
"""
Generate various plots to analyze model performance and data distributions.
"""

# 4a) Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         linestyle="--", color="red")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Optimized XGBoost: Actual vs. Predicted Prices")
plt.show()

# 4b) Boxplot: Property Price Distribution by Building Age
if "age_bucket" in df.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="age_bucket", y="price", data=df)
    plt.title("Property Price Distribution by Building Age")
    plt.xlabel("Building Age")
    plt.ylabel("Price (C$)")
    # Custom x-axis labels
    age_labels = ['0-10', '10-20', '20-30', '30-50', '50-100']
    plt.xticks(ticks=np.arange(5), labels=age_labels)
    plt.show()

# 4c) Violin Plot: Price Distribution by Building Age with Custom X-Axis Labels
if "age_bucket" in df.columns:
    plt.figure(figsize=(8, 6))
    sns.violinplot(x="age_bucket", y="price", data=df)
    plt.title("Violin Plot of Price by Building Age")
    plt.xlabel("Building Age")
    plt.ylabel("Price (C$)")

    # Custom x-axis labels
    age_labels = ['0-10', '10-20', '20-30', '30-50', '50-100']
    plt.xticks(ticks=np.arange(5), labels=age_labels)
    plt.show()

# 4d) Violin Plot: Price per Sqft by Building Age
if "age_bucket" in df.columns and "price_per_sqft" in df.columns:
    plt.figure(figsize=(8, 6))
    sns.violinplot(x="age_bucket", y="price_per_sqft", data=df)
    plt.title("Price per Sqft Distribution by Building Age")
    plt.xlabel("Building Age")
    plt.ylabel("Price per Sqft (C$/sqft)")

    # Custom x-axis labels
    age_labels = ['0-10', '10-20', '20-30', '30-50', '50-100']
    plt.xticks(ticks=np.arange(5), labels=age_labels)
    plt.show()

# 4e) Histogram: Overall Distribution of Property Prices
plt.figure(figsize=(8, 6))
sns.histplot(df["price"], kde=True, bins=30)
plt.title("Distribution of Property Prices")
plt.xlabel("Price (C$)")
plt.ylabel("Frequency")
plt.show()


# 4f) Residual Distribution Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel("Residuals (Actual - Predicted)")
plt.title("Residual Distribution")
plt.show()

# 4g) Feature Importance Plot from Best XGBoost Model
feature_names = X.columns  # X was defined before scaling
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance from XGBoost")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# 1) Identify top N features by importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 6  # pick how many you want to show
top_features = [feature_names[i] for i in indices[:top_n]]

# 2) Generate partial dependence plots only for those top features
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(
    best_model,
    X_scaled,
    features=top_features,
    feature_names=feature_names,
    ax=ax
)
plt.suptitle(f"Partial Dependence Plots for Top {top_n} Features")
plt.tight_layout()
plt.show()
