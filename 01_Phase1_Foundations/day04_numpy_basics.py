"""
Day 4: NumPy Basics for Machine Learning

Focus:
- NumPy arrays
- Shapes
- Basic operations
- Why NumPy is used in ML

No ML models.
No sklearn.
No shortcuts.
"""

import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6]])

print("1D array:", arr1)
print("2D array:\n", arr2)

# Array shapes
print("\nShape of arr1:", arr1.shape)
print("Shape of arr2:", arr2.shape)

# Basic operations
print("\nAddition:", arr1 + 2)
print("Multiplication:", arr1 * 2)

# Element-wise operations
arr3 = np.array([10, 20, 30, 40, 50])
print("\nElement-wise addition:", arr1 + arr3)

# Indexing and slicing
print("\nFirst element:", arr1[0])
print("Slice (index 1 to 3):", arr1[1:4])

# Observations:
# - NumPy arrays are faster than Python lists
# - Operations are vectorized
# - Shapes matter in ML computations
