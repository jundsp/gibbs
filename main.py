#%%
import numpy as np
import gibbs
import matplotlib.pyplot as plt

#%%
dp = gibbs.DirichletProcess(alpha=10)
z = []
for n in range(1000):
    z.append(dp.sample())
    
plt.plot(z), plt.xlabel('sample'), plt.ylabel('category'),plt.title('dirichlet process')
# %%
