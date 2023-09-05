#%%
from gibbs import Gibbs, Data, HMM
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('gibbs.mplstyles.latex')

#%%
# np.random.seed(123)

delta = np.array([0])
pi = np.ones(2) / 2
C = np.eye(2) + 1e-6
C[0,1] = 1e-1
C /= C.sum(-1)[:,None]
pz = pi

likelihood = np.ones(2)
for n in range(delta.shape[0]):
    likelihood *= C[delta[n]]

joint = likelihood * pi
marg = joint.sum()
post = joint / marg
plt.stem(post)
plt.ylim(0,1.1)
plt.xlim(-.1,1.1)

T = 100
N = 4
delta = np.random.randint(0,2,(T,N))

logl = np.log(C[delta]).sum(1)
model = HMM(states=2,expected_duration=1)
sampler = Gibbs()

for sample in range(10):
    model.forward(logl=logl)
    sampler.step(model.named_parameters())

sampler.get_estimates()
z = sampler._estimates['z']
print(delta)
print(z)
# %%
plt.imshow(np.concatenate([delta,z[:,None]],-1).T)
# %%
