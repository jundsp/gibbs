'''
Example of collapsed Gibbs sampling from Dirichlet Process infinite Gaussian mixture model.

Author: Julian Neri
Date: May 31, 2022
'''
#%%
import gibbs

#%%
gibbs.np.random.seed(1)
model = gibbs.DP_GMM(output_dim=2,alpha=2)
x = model.generate(200)[0]

# %%
model = gibbs.DP_GMM(output_dim=2,alpha=1,learn=True,outliers=True)

#%%
model.fit(x,samples=30)
model.plot()
model.plot_samples()

# %%
