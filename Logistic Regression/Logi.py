import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel(r"C:\Users\anarg\Downloads\LogReg dataset1.xlsx")

# Set display options
pd.set_option('display.width', None)

# Calculate probabilities and odds
df['Probability of Cured'] = df['Cured'] / df['Total Patients']
df['Probability of Not Cured'] = 1 - df['Probability of Cured']
df['Odds'] = df['Probability of Cured'] / df['Probability of Not Cured']
df['logit'] = np.log(df['Odds'])
print(df)

# Prepare data for regression
X = df['Medication Dosage'].values
Y = df['logit'].values

print("X (Medication Dosage):", X)
print("Y (Log-Odds):", Y)

# Calculate coefficients for logistic regression
mean_x = np.mean(X)
mean_y = np.mean(Y)

beta_1 = (np.sum(X * Y) - len(X) * mean_x * mean_y) / (np.sum(X**2) - len(X) * mean_x**2)
beta_0 = mean_y - beta_1 * mean_x

print(f"Intercept (beta_0): {beta_0}")
print(f"Slope (beta_1): {beta_1}")

# Calculate predicted log-odds and probabilities
y_pred = beta_0 + beta_1 * X
logi = np.exp(y_pred)
p_cure = logi / (1 + logi)

print("Predicted Probabilities of Cured:", p_cure)

# Predict on new unknown data
new_data = np.array([15, 24, 38, 60, 55])  # Example new dosages
new_y_pred = beta_0 + beta_1 * new_data
new_p_cure = 1 / (1 + np.exp(-new_y_pred))  # Convert log-odds to probabilities

# Apply threshold for classification
threshold = 0.5
predictions = (new_p_cure >= threshold).astype(int)  # 1 if cured, 0 if not cured

print("New Data Dosages:", new_data)
print("Predicted Probabilities for New Data:", new_p_cure)
print("Predicted Classes for New Data (1=Cured, 0=Not Cured):", predictions)

# Plot predictions and logistic curve for new data
plt.figure(figsize=(10, 6))

# Plot actual points for new data
for i in range(len(new_data)):
    if predictions[i] == 1:
        plt.scatter(new_data[i], new_p_cure[i], color='blue', label='Predicted Cured' if 'Predicted Cured' not in plt.gca().get_legend_handles_labels()[1] else "")
    else:
        plt.scatter(new_data[i], new_p_cure[i], color='orange', label='Predicted Not Cured' if 'Predicted Not Cured' not in plt.gca().get_legend_handles_labels()[1] else "")

# Plot logistic curve for new data
x_curve = np.linspace(min(new_data) - 10, max(new_data) + 10, 100)
y_curve = 1 / (1 + np.exp(-(beta_0 + beta_1 * x_curve)))
plt.plot(x_curve, y_curve, color='green', label='Logistic Curve')

# Plot threshold line
plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold (0.5)')

# Set plot labels and title
plt.xlabel('Medication Dosage')
plt.ylabel('Probability of Cured')
plt.title('Predictions and Logistic Curve for New Data')
plt.grid()
plt.legend()
plt.show()
