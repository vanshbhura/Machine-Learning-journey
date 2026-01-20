"""
Day 1: Statistics Foundations for Machine Learning

Focus:
- Mean
- Median
- Variance
- Standard Deviation
- Effect of outliers

No ML models.
No sklearn.
No shortcuts.
"""

import numpy as np

# Sample dataset
data = np.array([10, 12, 15, 18, 20, 100])

print("Data:", data)

# Mean
print("Mean:", data.mean())

# Median
print("Median:", np.median(data))

# Variance
print("Variance:", data.var())

# Standard Deviation
print("Standard Deviation:", data.std())

# Effect of outlier
clean_data = np.array([10, 12, 15, 18, 20])

print("\nAfter removing outlier:")
print("Clean data:", clean_data)
print("Mean:", clean_data.mean())
print("Median:", np.median(clean_data))
print("Variance:", clean_data.var())
print("Standard Deviation:", clean_data.std())
