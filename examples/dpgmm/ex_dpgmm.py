'''
Example of collapsed Gibbs sampling from Dirichlet Process infinite Gaussian mixture model.

Author: Julian Neri
Date: May 31, 2022
'''
#%%
import gibbs

#%%
model = gibbs.DP_GMM(output_dim=2,alpha=1)
x = model.generate(100)[0]
# %%
model = gibbs.DP_GMM(output_dim=2,alpha=1)
model.fit(x,samples=20)
model.plot()
model.plot_samples()

# %%
