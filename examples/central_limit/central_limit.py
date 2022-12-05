import numpy as np
import seaborn as sns

events = int(1e4)
sample_size = 10

x = np.random.uniform(0,1,events)
x = x.reshape(-1,sample_size)

# The mean of sample set is distributed normally around mean (expected value) of sampling distribution.
xbar = x.mean(-1)
sns.histplot(xbar)

