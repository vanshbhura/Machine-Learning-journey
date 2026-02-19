"""
Day 15: Multiple Linear Regression (Concept + Implementation)

Focus:
- Multiple input features
- Vectorized predictions
- Weight vector instead of single slope
- Linear regression in higher dimensions

No sklearn.
Pure NumPy.
"""

import numpy as np

# Dataset
# Example: [size, bedrooms]
X = np.array([
    [500, 1],
    [800, 2],
    [1200, 3],
    [1600, 3],
    [2000, 4]
])

# Target: house prices
y = np.array([25000, 40000, 65000, 90000, 120000])

# Initialize weights and bias
weights = np.zeros(X.shape[1])
bias = 0

learning_rate = 0.00000001
epochs = 1000
n = len(X)

for i in range(epochs):

    # Predictions: y = Xw + b
    y_pred = np.dot(X, weights) + bias

    # Errors
    error = y - y_pred

    # Gradients
    dw = (-2/n) * np.dot(X.T, error)
    db = (-2/n) * np.sum(error)

    # Update
    weights -= learning_rate * dw
    bias -= learning_rate * db

    if (i + 1) % 200 == 0:
        cost = np.mean(error ** 2)
        print(f"Epoch {i+1}: cost = {cost:.2f}")

print("\nFinal weights:", weights)
print("Final bias:", bias)

# Predictions
predictions = np.dot(X, weights) + bias
print("\nPredictions:", predictions)

# Observations:
# - Multiple features require weight vector
# - Dot product handles multi-dimensional data
# - Same gradient descent logic applies
