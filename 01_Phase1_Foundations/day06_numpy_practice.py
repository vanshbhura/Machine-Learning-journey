"""
Day 6: NumPy Practice and Revision

Focus:
- Array creation
- Vector operations
- Mean normalization
- Practice without loops

No ML models.
No sklearn.
No shortcuts.
"""

import numpy as np

# Create data
data = np.array([5, 10, 15, 20, 25])

print("Data:", data)

# Basic operations
print("\nAdd 5:", data + 5)
print("Multiply by 2:", data * 2)

# Vector subtraction
mean_value = data.mean()
print("\nMean:", mean_value)

normalized = data - mean_value
print("Mean normalized data:", normalized)

# Square each element
squared = data ** 2
print("\nSquared values:", squared)

# 2D array practice
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print("\nMatrix:\n", matrix)
print("Matrix shape:", matrix.shape)

# Column-wise mean
col_mean = matrix.mean(axis=0)
print("Column-wise mean:", col_mean)

# Row-wise mean
row_mean = matrix.mean(axis=1)
print("Row-wise mean:", row_mean)

# Observations:
# - NumPy replaces loops with vectorized operations
# - Axis matters in matrix operations
# - Mean normalization is common in ML
