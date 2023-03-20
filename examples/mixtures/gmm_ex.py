#%%
from gibbs import Gibbs, InfiniteGMM, GMM, gmm_generate, plot_cov_ellipse, get_colors, get_scatter_kwds,scattercat,Data
import numpy as np
import os
import matplotlib.pyplot as plt
from variational import VB_GMM


np.random.seed(123)
y = gmm_generate(500,2,5)[0]
data = Data(y=y)
data.plot()

#%%
np.random.seed(123)
model = VB_GMM(output_dim=2,n_components=8)

#%%
model.fit(data.output,epochs=10)
model.plot(**get_scatter_kwds())

#%%
np.random.seed(123)
model = GMM(components=8,hyper_sample=True)
sampler = Gibbs()

#%%
sampler.fit(data,model,samples=100)

z_hat = sampler._estimates['mix.z'].astype(int)
colors = get_colors()
scattercat(data.output,z_hat)
for k in np.unique(z_hat):
    mu,cov = sampler._estimates['theta.{}.A'.format(k)].ravel(),sampler._estimates['theta.{}.Q'.format(k)]
    plot_cov_ellipse(mu,cov,fill=None,color=colors[k])

#%%
modeli = InfiniteGMM()
sampleri = Gibbs()

np.random.seed(123)
#%%
sampleri.fit(data,modeli,samples=20) 

sampleri.get_estimates(burn_rate=.75)
#%%
z_hat = modeli.z
scattercat(data.output,z_hat)
for k in np.unique(z_hat):
    idx = z_hat == k
    mu,S,nu = modeli._predictive_parameters(*modeli._posterior(modeli.y[idx],*modeli.theta))
    cov = S * (nu)/(nu-2)
    plot_cov_ellipse(mu,cov,fill=None,color=colors[k])

chain = sampleri.get_chain(burn_rate=0,flatten=False)
fig,ax = plt.subplots(len(chain),figsize=(5,1.5*len(chain)))
for ii,p in enumerate(chain):
    _x = chain[p]
    _x = _x.reshape(_x.shape[0],-1)
    ax[ii].plot(_x,'k',alpha=.1)
    ax[ii].set_title(p)
plt.tight_layout()

# %%