import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\anarg\OneDrive\Documents\Salary_Data.csv")

X = df.iloc[:,:-1].values.flatten()

Y=df.iloc[:,-1].values

mean_x = np.mean(X)
mean_y = np.mean(Y)

beta_1 = (np.sum(X*Y) - len(X) * mean_x * mean_y) / (np.sum(X**2) - len(X) * mean_x**2)
beta_0 = mean_y - beta_1 * mean_x

print(f"Intercept (beta_1): {beta_0}")
print(f"Slope (beta_0): {beta_1}")

plt.scatter(X, Y, color='blue', label = 'Data Points')

Y_pred =  beta_0 + beta_1  * X

plt.plot(X, Y_pred, color='red', label = 'Regression Line')

plt.xlabel('X - Independent Variable')
plt.ylabel('Y - Dependent Variable')
plt.title('Linear Regression')
plt.legend()
plt.show()