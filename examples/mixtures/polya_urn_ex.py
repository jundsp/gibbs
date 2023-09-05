import numpy as np
from scipy.stats import multinomial
import matplotlib.pyplot as plt

plt.style.use('gibbs.mplstyles.latex')

np.random.seed(1)
N = 200
n_trials = 3*3

X = []
for i in range(n_trials):
    K = np.array([1,1])*(i+1)
    pi = K / K.sum()
    x = np.zeros((N,2))
    for n in range(N):
        temp = multinomial.rvs(1,pi).argmax()
        K[temp] += 1
        pi = K / K.sum()
        x[n] = K.copy()
    X.append(x)
X = np.stack(X,0)
fig,ax = plt.subplots(3,3,figsize=(6,6),sharex=True,sharey=True)
ax = ax.ravel()
for i in range(n_trials):
    ax[i].plot(X[i,:,0],c='b')
    ax[i].plot(X[i,:,1],c='r')
    ax[i].set_xlim(0,N)
    ax[i].set_ylim(0)
plt.tight_layout()