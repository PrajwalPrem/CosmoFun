###########################        LINEAR REGRESSION (Figure 1)              ###########################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate some sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Perform linear regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
best_fit_line = lin_reg.predict(X)


# Plotting the original data points
plt.scatter(X, y, color='blue', label='Data Points')


# Plotting the best-fit line from linear regression
plt.plot(X, best_fit_line, color='red', label='Best-Fit Line (Linear Regression)')

plt.title('Different Lines Representing the Same Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()




###########################        Comparing LINEAR, MULTILINEAR and POLYNOMIAL REGRESSION (Figure 2)              ###########################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
X1 = 2 * np.random.rand(100, 1)
X2 = 1.5 * np.random.rand(100, 1)
y = 4 + 3 * X1 + 2 * X2 + 1.5 * X1 * X2 + np.random.randn(100, 1)

# Split the data into training and testing sets
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X1_train, y_train)
y_pred_linear = linear_model.predict(X1_test)

# Multiple Linear Regression
multiple_linear_model = LinearRegression()
# Concatenate X1 and X2 to form the input for multiple linear regression
multiple_linear_model.fit(np.concatenate([X1_train, X2_train], axis=1), y_train)
y_pred_multiple_linear = multiple_linear_model.predict(np.concatenate([X1_test, X2_test], axis=1))

# Polynomial Regression
degree = 2  # Degree of the polynomial
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
# Transform the features to include polynomial features
X_poly_train = poly_features.fit_transform(np.concatenate([X1_train, X2_train], axis=1))
X_poly_test = poly_features.transform(np.concatenate([X1_test, X2_test], axis=1))

polynomial_model = LinearRegression()
polynomial_model.fit(X_poly_train, y_train)
y_pred_polynomial = polynomial_model.predict(X_poly_test)

# Evaluate the models
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_multiple_linear = mean_squared_error(y_test, y_pred_multiple_linear)
mse_polynomial = mean_squared_error(y_test, y_pred_polynomial)

# Print final expressions
print(f"Linear Regression: y = {linear_model.coef_[0][0]:.2f} * X1 + {linear_model.intercept_[0]:.2f}")
print(f"Multiple Linear Regression: y = {multiple_linear_model.coef_[0][0]:.2f} * X1 + {multiple_linear_model.coef_[0][1]:.2f} * X2 + {multiple_linear_model.intercept_[0]:.2f}")
print(f"Polynomial Regression (Degree {degree}): y = {polynomial_model.intercept_[0]:.2f} + {polynomial_model.coef_[0][0]:.2f} * X1 + {polynomial_model.coef_[0][1]:.2f} * X2 + {polynomial_model.coef_[0][2]:.2f} * X1^2 + {polynomial_model.coef_[0][3]:.2f} * X1 * X2 + {polynomial_model.coef_[0][4]:.2f} * X2^2")

# Plotting the results
plt.figure(figsize=(15, 5))

# Linear Regression Plot
plt.subplot(1, 3, 1)
plt.scatter(X1_test, y_test, color='blue', label='Actual')
plt.plot(X1_test, y_pred_linear, color='red', linewidth=2, label=f'Predicted\n{linear_model.coef_[0][0]:.2f} * X1 + {linear_model.intercept_[0]:.2f}')
plt.title(f'Linear Regression\nMSE: {mse_linear:.2f}')
plt.xlabel('X1')
plt.ylabel('y')
plt.legend()

# Multiple Linear Regression Plot
plt.subplot(1, 3, 2)
plt.scatter(X1_test, y_test, color='blue', label='Actual')
plt.scatter(X1_test, y_pred_multiple_linear, color='red', linewidth=2, label=f'Predicted\n{multiple_linear_model.coef_[0][0]:.2f} * X1 + {multiple_linear_model.coef_[0][1]:.2f} * X2 + {multiple_linear_model.intercept_[0]:.2f}')
plt.title(f'Multiple Linear Regression\nMSE: {mse_multiple_linear:.2f}')
plt.xlabel('X1')
plt.ylabel('y')
plt.legend()

# Polynomial Regression Plot
plt.subplot(1, 3, 3)
plt.scatter(X1_test, y_test, color='blue', label='Actual')
sorted_zip = sorted(zip(X1_test, y_pred_polynomial))
X1_test_sorted, y_pred_polynomial_sorted = zip(*sorted_zip)
plt.plot(X1_test_sorted, y_pred_polynomial_sorted, color='red', linewidth=2, label=f'Predicted\n{polynomial_model.intercept_[0]:.2f} + {polynomial_model.coef_[0][0]:.2f} * X1 + {polynomial_model.coef_[0][1]:.2f} * X2 + {polynomial_model.coef_[0][2]:.2f} * X1^2 + {polynomial_model.coef_[0][3]:.2f} * X1 * X2 + {polynomial_model.coef_[0][4]:.2f} * X2^2')
plt.title(f'Polynomial Regression (Degree {degree})\nMSE: {mse_polynomial:.2f}')
plt.xlabel('X1')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()













































