# Real Estate Price Prediction

## Overview
This project builds an **XGBoost regression model** to predict real estate property prices based on various features. The dataset is preprocessed, and the model is trained using hyperparameter tuning to improve prediction accuracy. Multiple **visualizations** and **prediction examples** are also included.

## Features & Workflow
### 1️⃣ Data Loading
- Loads **real-estate-data.csv** into a Pandas DataFrame.

### 2️⃣ Data Preprocessing
- Converts **size ranges** (e.g., "500-999 sqft") into numeric averages.
- Handles **missing values** with median imputation.
- **Encodes categorical features** (e.g., one-hot encoding for `ward` and `exposure`).
- Creates **new features** like `price_per_sqft` and `age_bucket`.

### 3️⃣ Model Training
- Splits data into **training (80%)** and **testing (20%)** sets.
- Scales features using `StandardScaler`.
- Trains **XGBoost Regressor** with `GridSearchCV` for hyperparameter tuning.
- Evaluates model using **R² score, MAE, and MSE**.

### 4️⃣ Data Visualization
- **Scatter plot:** Actual vs. Predicted Prices
- **Boxplot & Violin plots:** Property price distribution by age
- **Histogram:** Property price distribution
- **Residuals plot:** Model errors
- **Feature importance plot:** Key factors influencing price

### 5️⃣ Predicting New Properties
- Predicts property prices for **new sample properties** using the trained model.

## Installation & Dependencies
Make sure you have the following installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## How to Run
1. Place `real-estate-data.csv` in the working directory.
2. Run the script:
   ```bash
   python real_estate_price_prediction.py
   ```
3. View model performance metrics and generated visualizations.

## Example Prediction Output
```
Predicted Price for Property 1: $750,000.00
Predicted Price for Property 2: $520,000.00
Predicted Price for Property 3: $1,100,000.00
```

## Future Improvements
- Add **more features** (e.g., location-based attributes).
- Improve hyperparameter tuning with **Bayesian optimization**.
- Experiment with **deep learning models** for price prediction.

## License
This project is licensed under the **MIT License**.


