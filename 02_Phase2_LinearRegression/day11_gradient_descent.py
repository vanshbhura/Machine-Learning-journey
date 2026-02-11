"""
Day 11: Gradient Descent Intuition

Focus:
- How parameters update
- Reducing cost step by step
- Learning rate
- Basic gradient descent loop

No sklearn.
No advanced optimization.
Only core idea.
"""

import numpy as np

# Data
X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

# Initialize parameters
m = 0
b = 0

learning_rate = 0.01
epochs = 20
n = len(X)

print("Initial m:", m)
print("Initial b:", b)

for i in range(epochs):

    # Predictions
    y_pred = m * X + b

    # Errors
    error = y - y_pred

    # Gradients
    dm = (-2/n) * np.sum(X * error)
    db = (-2/n) * np.sum(error)

    # Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

    # Cost (MSE)
    cost = np.mean(error ** 2)

    print(f"Epoch {i+1}: m={m:.4f}, b={b:.4f}, cost={cost:.4f}")

print("\nFinal parameters:")
print("m:", m)
print("b:", b)

# Observations:
# - Gradient descent updates parameters step by step
# - Learning rate controls update size
# - Cost should decrease over time
# - Goal is to minimize error
