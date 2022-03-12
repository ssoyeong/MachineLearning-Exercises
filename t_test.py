import numpy as np
from scipy import stats

# Sample Size
N = 10

# Dataset
older = [45, 38, 52, 48, 25, 39, 51, 46, 55, 46]
younger = [34, 22, 15, 27, 37, 41, 24, 19, 26, 36]

# Calculate the Variance and the Standard Deviation
var_older = np.var(older, ddof=1)
var_younger = np.var(younger, ddof=1)

s = np.sqrt((var_older + var_younger)/2)

# Calculate the t-statistics
t = (np.mean(older) - np.mean(younger))/(s*np.sqrt(2/N))

# Compare with the critical t value
# Degrees of freedom
df = 2 * N - 2
# p value after comparision with the t
p = 1 - stats.t.cdf(t, df=df)

print("t = " + str(t))
print("p = " + str(2*p))

# Cross check with the internal SciPy function
# Use scipy.stats.ttest_rel() for paired-samples test
t2, p2 = stats.ttest_ind(older, younger)
print("\nt = " + str(t2))
print("p = " + str(p2))


