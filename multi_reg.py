import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

n = 60  

area_sqft = np.random.randint(500, 3000, n)
num_bedrooms = np.random.randint(1, 5, n)
age_years = np.random.randint(0, 30, n)

#Creating target with some relation and noise
price_lakhs = (
    area_sqft * 0.05 +
    num_bedrooms * 5 -
    age_years * 0.3 +
    np.random.normal(0, 5, n)
)

data = pd.DataFrame({
    'area_sqft': area_sqft,
    'num_bedrooms': num_bedrooms,
    'age_years': age_years,
    'price_lakhs': price_lakhs
})

print("First 5 rows of dataset:\n", data.head())

#TASK 1
X = data[['area_sqft', 'num_bedrooms', 'age_years']]
y = data['price_lakhs']

model = LinearRegression()
model.fit(X, y)

print("\nIntercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

y_pred = model.predict(X)

print("\nActual vs Predicted (first 5):")
for actual, pred in list(zip(y, y_pred))[:5]:
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

#TASK 2
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("\nEvaluation Metrics:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

"""
MAE shows the average prediction error
RMSE penalizes larger errors more than MAE
R2 indicates how well the model explains the variance(closer to 1 is better)
"""

#TASK 3
residuals = y - y_pred

plt.figure()
plt.hist(residuals, bins=10)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

#If residuals are normally distributed around 0, the model is good.
