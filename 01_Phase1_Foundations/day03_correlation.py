"""
Day 3: Correlation Basics

Focus:
- Understanding correlation between variables
- Positive and negative correlation
- Why correlation does not mean causation

No ML models.
No sklearn.
No shortcuts.
"""

import numpy as np

# Two related variables
hours_studied = np.array([1, 2, 3, 4, 5])
scores = np.array([35, 45, 55, 65, 75])

print("Hours studied:", hours_studied)
print("Scores:", scores)

# Correlation coefficient
correlation = np.corrcoef(hours_studied, scores)[0, 1]
print("\nCorrelation coefficient:", correlation)

# Negative correlation example
sleep_hours = np.array([2, 4, 6, 8, 10])
screen_time = np.array([10, 8, 6, 4, 2])

neg_correlation = np.corrcoef(sleep_hours, screen_time)[0, 1]
print("\nNegative correlation coefficient:", neg_correlation)

# Correlation without causation example
ice_cream_sales = np.array([10, 20, 30, 40, 50])
temperature = np.array([20, 25, 30, 35, 40])

fake_correlation = np.corrcoef(ice_cream_sales, temperature)[0, 1]
print("\nIce cream sales vs temperature correlation:", fake_correlation)

# Observations:
# - Correlation measures linear relationship
# - Positive correlation: both increase together
# - Negative correlation: one increases, other decreases
# - Correlation does not imply causation
