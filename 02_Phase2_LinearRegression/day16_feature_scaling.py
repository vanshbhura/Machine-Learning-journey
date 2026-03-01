"""
Day 16: Feature Scaling

Focus:
- Why scaling is needed
- Standardization
- Faster gradient descent convergence
- Prevent one feature dominating others

No sklearn.
Pure NumPy.
"""

import numpy as np

# Dataset: [size, bedrooms]
X = np.array([
    [500, 1],
    [800, 2],
    [1200, 3],
    [1600, 3],
    [2000, 4]
])

y = np.array([25000, 40000, 65000, 90000, 120000])

# -------------------------------
# Feature Scaling (Standardization)
# -------------------------------
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

X_scaled = (X - X_mean) / X_std

print("Scaled features:\n", X_scaled)

# -------------------------------
# Linear Regression with scaled data
# -------------------------------
weights = np.zeros(X.shape[1])
bias = 0

learning_rate = 0.01
epochs = 1000
n = len(X_scaled)

for i in range(epochs):

    y_pred = np.dot(X_scaled, weights) + bias
    error = y - y_pred

    dw = (-2/n) * np.dot(X_scaled.T, error)
    db = (-2/n) * np.sum(error)

    weights -= learning_rate * dw
    bias -= learning_rate * db

    if (i + 1) % 200 == 0:
        cost = np.mean(error ** 2)
        print(f"Epoch {i+1}: cost = {cost:.2f}")

print("\nFinal weights:", weights)
print("Final bias:", bias)

# Observations:
# - Scaling improves convergence
# - Learning rate can be larger
# - Features contribute more evenly
