import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
file_path = "/Users/franklinlu/PycharmProjects/PythonProject/.venv/real-estate-data.csv"  # Update with the correct path if needed
df = pd.read_csv(file_path)

# Convert 'size' from categorical range to numeric (taking the average of the range)
def convert_size(size):
    if isinstance(size, str) and "sqft" in size:
        size_range = size.replace(" sqft", "").split("-")
        if len(size_range) == 2:
            return (int(size_range[0]) + int(size_range[1])) / 2
    return None

df["size"] = df["size"].apply(convert_size)

# Fill missing values
df.fillna({"beds": df["beds"].median(), "size": df["size"].median(), "maint": df["maint"].median(), "D_mkt": df["D_mkt"].median()}, inplace=True)
df.dropna(subset=["price"], inplace=True)  # Drop rows with missing price (target variable)

# Encode categorical variables
df["DEN"] = df["DEN"].map({"YES": 1, "No": 0})
df["parking"] = df["parking"].map({"Yes": 1, "N": 0})
df = pd.get_dummies(df, columns=["ward", "exposure"], drop_first=True)

# Drop unnecessary columns
df.drop(columns=["id_"], inplace=True)

# Define features and target variable
X = df.drop(columns=["price"])
y = df["price"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Example new property data (update values accordingly)
new_property = {
    "beds": 3,
    "baths": 2,
    "size": 1200,  # in sqft
    "parking": 1,  # 1 for Yes, 0 for No
    "DEN": 0,  # 1 for Yes, 0 for No
    "maint": 500,
    "D_mkt": 30,
    "building_age": 10,
    "ward_W2": 0,  # Ensure correct one-hot encoding
    "ward_W3": 1,
    "exposure_S": 0,
    "exposure_W": 1,
}


# Convert dictionary to DataFrame
new_property_df = pd.DataFrame([new_property])

# Ensure all columns match the training set (fill missing ones with 0)
new_property_df = new_property_df.reindex(columns=X.columns, fill_value=0)

# Predict price
predicted_price = model.predict(new_property_df)

print(f"Estimated Price: ${predicted_price[0]:,.2f}")




# Scatter plot: Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red")  # 45-degree reference line

# Labels and Title
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted Real Estate Prices")
plt.show()
