##################################### The First Example #####################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from gplearn.genetic import SymbolicRegressor

# Generate synthetic data for a complicated curve
np.random.seed(42)

# Define a complicated curve (you can replace this with any function)
def true_curve(x):
    return np.sin(x) + np.cos(2 * x) + np.random.normal(0, 0.5, size=len(x))

# Generate x values
x_values = np.linspace(0, 6, 200)
# Generate y values with some noise
y_values = true_curve(x_values)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_values.reshape(-1, 1), y_values, test_size=0.2, random_state=42
)

# Plot the synthetic data
plt.scatter(x_values, y_values, label='Measured Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Synthetic Data for a Complicated Curve')
plt.legend()
plt.show()

# Use Random Forest regression to fit an equation
regressor = RandomForestRegressor(n_estimators=200, random_state=42)
regressor.fit(x_train, y_train)
y_pred_rf = regressor.predict(x_test)

# Plot the Random Forest regression result
plt.scatter(x_test, y_test, label='Measured Data')
plt.plot(x_test, y_pred_rf, color='green', linestyle='dashed', label='Random Forest Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Forest Regression on Complicated Data')
plt.legend()
plt.show()


symreg = SymbolicRegressor(generations=100, population_size=500,
                           function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos'),
                           random_state=42)
symreg.fit(x_train, y_train)

# Predict with Symbolic Regression
y_pred_symreg = symreg.predict(x_test)

# Get the symbolic regression equation
expression = str(symreg._program)
print(f"Symbolic Regression Equation: {expression}")

# Evaluate the symbolic expression on a range of x values to create a curve
x_curve = np.linspace(0, 6, 200)
y_curve = [symreg.predict(np.array([[x]]))[0] for x in x_curve]

# Plot the Symbolic Regression result
plt.scatter(x_test, y_test, label='Measured Data')
plt.plot(x_curve, y_curve, color='purple', linestyle='dashed', label='Symbolic Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Symbolic Regression on Synthetic Data')
plt.legend()
plt.show()





##################################### Hubble Function #####################################
##################################### Makes use of the data (File name: H(z)data.dat) in the same repository #####################################
import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor

# Load real observational data from the file H(z)data.dat
data = np.loadtxt('H(z)data.dat', delimiter='\t')

# Separate the data into redshift (z), Hubble function (H(z)), and error bar values (σ)
z_values = data[:, 0].reshape(-1, 1)
H_values = data[:, 1]
sigma_values = data[:, 2]

# Perform symbolic regression using 'log' to approximate exponential behavior
symreg = SymbolicRegressor(generations=100, population_size=500,
                           function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'log'),
                           random_state=42)
symreg.fit(z_values, H_values)

# Get the symbolic regression equation
expression = str(symreg._program)
print(f"Symbolic Regression Equation: {expression}")

# Predict H(z) values using the symbolic expression
H_predicted = symreg.predict(z_values)

# Plot the data points, error bars, and the symbolic regression result
plt.figure(figsize=(10, 6))
plt.errorbar(z_values, H_values, yerr=sigma_values, label='Measured Data', color='blue', fmt='o', markersize=5, capsize=3)
plt.plot(z_values, H_predicted, color='red', label='Symbolic Regression', linewidth=2)
plt.xlabel('Redshift z')
plt.ylabel('Hubble Function H(z)')
plt.title('Symbolic Regression of the Hubble Function')
plt.legend()
plt.grid()
plt.show()






