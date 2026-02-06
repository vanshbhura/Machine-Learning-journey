"""
Day 9: Simple Linear Regression Intuition

Focus:
- Linear relationship between X and y
- Line equation: y = mx + b
- Understanding slope and intercept
- Making predictions using a line

No sklearn.
No training yet.
Only intuition using NumPy.
"""

import numpy as np

# Input feature (X)
X = np.array([1, 2, 3, 4, 5])

# Target output (y)
y = np.array([3, 5, 7, 9, 11])

print("X:", X)
print("y:", y)

# Assume slope (m) and intercept (b)
m = 2
b = 1

print("\nAssumed line: y = m*x + b")
print("Slope (m):", m)
print("Intercept (b):", b)

# Predictions using the line
y_pred = m * X + b
print("\nPredicted y:", y_pred)

# Error (difference between actual and predicted)
error = y - y_pred
print("Error:", error)

# Mean error
mean_error = error.mean()
print("Mean error:", mean_error)

# Predict for new value
new_x = 6
new_y = m * new_x + b
print("\nPrediction for x = 6:", new_y)

# Observations:
# - Linear regression fits a straight line
# - Slope controls steepness
# - Intercept shifts the line
# - Error tells how good the line is
