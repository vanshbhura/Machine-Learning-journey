"""
Day 7: Revision and Small Statistics Practice

Focus:
- Revising NumPy basics
- Revising statistics concepts
- Applying everything together
- No new concepts

No ML models.
No sklearn.
No shortcuts.
"""

import numpy as np

# Dataset for practice
data = np.array([2, 4, 6, 8, 10, 50])

print("Data:", data)

# Basic statistics
mean_value = data.mean()
median_value = np.median(data)
variance_value = data.var()
std_value = data.std()

print("\nStatistics:")
print("Mean:", mean_value)
print("Median:", median_value)
print("Variance:", variance_value)
print("Standard Deviation:", std_value)

# Detecting outlier manually
print("\nChecking effect of outlier:")
clean_data = data[data < 20]

print("Clean data:", clean_data)
print("Mean (clean):", clean_data.mean())
print("Median (clean):", np.median(clean_data))
print("Std (clean):", clean_data.std())

# NumPy operations revision
print("\nNumPy operations:")
print("Data * 2:", data * 2)
print("Data + 5:", data + 5)

# Mean normalization
normalized_data = data - mean_value
print("\nMean normalized data:", normalized_data)

# Observations:
# - Outliers heavily affect mean and std
# - Median is more stable
# - NumPy enables fast vectorized operations
# - Strong foundations are required before ML
