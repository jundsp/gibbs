import numpy as np
from scipy.stats import multinomial
import matplotlib.pyplot as plt

plt.style.use('gibbs.mplstyles.latex')

np.random.seed(1)
K = np.array([1,1])*10
pi = K / K.sum()
N = 200

x = np.zeros((N,2))
for n in range(N):
    temp = multinomial.rvs(1,pi).argmax()
    K[temp] += 1
    pi = K / K.sum()
    x[n] = K.copy()

plt.figure(figsize=(3,2.5))
plt.plot(x[:,0],c='b')
plt.plot(x[:,1],c='r')
plt.xlim(0,N)
plt.ylim(0)