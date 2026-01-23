"""
Day 2: Data Distributions and Outliers

Focus:
- Understanding data distribution
- Detecting outliers
- How outliers affect statistics

No ML models.
No sklearn.
No shortcuts.
"""

import numpy as np

# Dataset with normal values
data = np.array([10, 12, 15, 18, 20, 22, 25])

print("Original data:", data)

# Basic statistics
print("Mean:", data.mean())
print("Median:", np.median(data))
print("Standard Deviation:", data.std())

# Adding an outlier
data_with_outlier = np.append(data, 100)

print("\nData with outlier:", data_with_outlier)

print("Mean with outlier:", data_with_outlier.mean())
print("Median with outlier:", np.median(data_with_outlier))
print("Standard Deviation with outlier:", data_with_outlier.std())

# Simple outlier detection using threshold
threshold = data_with_outlier.mean() + 2 * data_with_outlier.std()

outliers = data_with_outlier[data_with_outlier > threshold]

print("\nDetected outliers:", outliers)

# Observations:
# - Mean increases due to outlier
# - Median stays relatively stable
# - Std deviation increases significantly
# - Outliers distort data distribution
