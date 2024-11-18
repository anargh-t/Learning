import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# XOR input and output
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

# Set seed for reproducibility
np.random.seed(42)

# Network structure
input_layer_neurons = 2   # Input layer (2 inputs)
hidden_layer_neurons = 2  # Hidden layer (2 neurons)
output_neurons = 1        # Output layer (1 output)

# Initialize weights with random values
hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
output_weights = np.random.uniform(size=(hidden_layer_neurons, output_neurons))

# Learning rate
learning_rate = 0.5

# Number of epochs for training
epochs = 10000

# Track error for each epoch
error_list = []

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, hidden_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, output_weights)
    predicted_output = sigmoid(output_layer_input)

    # Calculate error
    error = y - predicted_output
    mean_squared_error = np.mean(np.square(error))
    error_list.append(mean_squared_error)  # Track error

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    hidden_weights += X.T.dot(d_hidden_layer) * learning_rate

# Output after training
print("Output after training:")
print(predicted_output)

# Plotting the error over epochs
plt.plot(error_list)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Training Error Over Time")
plt.show()

# Testing on the XOR input
print("\nTesting on XOR inputs:")
for i in range(len(X)):
    hidden_layer_input = np.dot(X[i], hidden_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, output_weights)
    predicted_output = sigmoid(output_layer_input)

    print(f"Input: {X[i]} -> Predicted Output: {predicted_output.round()} (Raw Output: {predicted_output})")
