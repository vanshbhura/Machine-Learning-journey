"""
Day 12: Linear Regression from Scratch

Focus:
- Combine prediction, cost, and gradient descent
- Train model properly
- Final parameters after training

No sklearn.
Pure NumPy.
"""

import numpy as np

# Dataset
X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

# Initialize parameters
m = 0
b = 0

learning_rate = 0.01
epochs = 100
n = len(X)

# Training loop
for i in range(epochs):

    # Predictions
    y_pred = m * X + b

    # Errors
    error = y - y_pred

    # Cost (MSE)
    cost = np.mean(error ** 2)

    # Gradients
    dm = (-2/n) * np.sum(X * error)
    db = (-2/n) * np.sum(error)

    # Update parameters
    m -= learning_rate * dm
    b -= learning_rate * db

    if (i + 1) % 10 == 0:
        print(f"Epoch {i+1}: cost = {cost:.4f}")

print("\nFinal parameters:")
print("m:", round(m, 4))
print("b:", round(b, 4))

# Final predictions
final_predictions = m * X + b
print("\nFinal predictions:", final_predictions)

# Observations:
# - Model learns slope and intercept
# - Cost decreases over time
# - Gradient descent finds best fitting line
