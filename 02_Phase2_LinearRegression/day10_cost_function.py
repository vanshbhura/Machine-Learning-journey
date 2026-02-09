"""
Day 10: Cost Function (Error Measurement)

Focus:
- Why error exists
- Measuring how bad predictions are
- Mean Squared Error (MSE)
- Cost value for a regression line

No sklearn.
No optimization yet.
Only error calculation.
"""

import numpy as np

# Input feature
X = np.array([1, 2, 3, 4, 5])

# Actual output
y = np.array([3, 5, 7, 9, 11])

# Assumed parameters (same as Day 9)
m = 2
b = 1

# Predictions
y_pred = m * X + b

print("Actual y:", y)
print("Predicted y:", y_pred)

# Error
error = y - y_pred
print("\nError:", error)

# Squared error
squared_error = error ** 2
print("Squared error:", squared_error)

# Mean Squared Error (Cost Function)
mse = squared_error.mean()
print("\nMean Squared Error (Cost):", mse)

# Trying worse parameters
m_bad = 1
b_bad = 0

y_pred_bad = m_bad * X + b_bad
error_bad = y - y_pred_bad
mse_bad = (error_bad ** 2).mean()

print("\nWith worse parameters:")
print("Predicted y:", y_pred_bad)
print("Cost:", mse_bad)

# Observations:
# - Cost function measures prediction quality
# - Lower cost means better fit
# - Squaring penalizes large errors
# - Training aims to minimize this cost
