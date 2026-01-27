"""
Day 5: NumPy Advanced Operations

Focus:
- Dot product
- Broadcasting
- Vector math used in ML

No ML models.
No sklearn.
No shortcuts.
"""

import numpy as np

# Vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Vector a:", a)
print("Vector b:", b)

# Dot product
dot_product = np.dot(a, b)
print("\nDot product:", dot_product)

# Matrix multiplication
matrix1 = np.array([[1, 2],
                     [3, 4]])
matrix2 = np.array([[5, 6],
                     [7, 8]])

print("\nMatrix multiplication:\n", np.dot(matrix1, matrix2))

# Broadcasting
data = np.array([10, 20, 30, 40, 50])
mean = data.mean()

print("\nData:", data)
print("Mean:", mean)

# Subtracting mean from each element (broadcasting)
normalized_data = data - mean
print("Normalized data:", normalized_data)

# Observations:
# - Dot product is core to ML calculations
# - Broadcasting avoids explicit loops
# - Vectorized math makes ML efficient
