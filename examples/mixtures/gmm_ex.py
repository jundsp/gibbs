#%%
from gibbs import Gibbs, InfiniteGMM, GMM, gmm_generate, plot_cov_ellipse, get_colors, get_scatter_kwds,scattercat, Data, relabel, categorical2multinomial
import numpy as np
import os
import matplotlib.pyplot as plt
from variational import VB_GMM

plt.style.use('gibbs.mplstyles.latex')

# * Compute times or complexity to add into the thesis
figsize=(3,2.5)
colors = get_colors()
# colors = np.array(['r','g','b','m','y','k','orange','grey']*30)
np.random.seed(123)
y, z = gmm_generate(500,2,5)
data = Data(y=y)
data.plot()

scattercat(data.output,z,figsize=figsize,colors=colors)
plt.savefig("imgs/gmm_data_ex.pdf")

#%%
np.random.seed(123)
model_vb = VB_GMM(output_dim=2,n_components=8)

#%%
model_vb.fit(data.output,epochs=200)
# %%
model_vb.plot(**get_scatter_kwds(),figsize=figsize,colors=colors)
plt.tight_layout()
plt.savefig("imgs/gmm_vb_ex.pdf")

plt.figure()
model_vb.plot_objectives()

plt.savefig("imgs/gmm_vb_optim.pdf")

#%%
np.random.seed(123)
model = GMM(components=8,hyper_sample=True)
sampler = Gibbs()

#%%
sampler.fit(data,model,samples=200)

#%%
chain = sampler.get_chain(burn_rate=.5)

tau = relabel(probs=chain['mix.rho'],verbose=True,iters=20)

rho = chain['mix.rho']
rho = np.take_along_axis(rho,tau[:,None,:],-1)
z_hat = rho.mean(0).argmax(-1)
z_unique = np.unique(z_hat)
for i, k in enumerate(z_unique):
    z_hat[z_hat==k] = i
scattercat(data.output,z_hat,figsize=figsize,colors=colors)
for i,k in enumerate(z_unique):
    mu,cov = sampler._estimates['theta.{}.A'.format(k)].ravel(),sampler._estimates['theta.{}.Q'.format(k)]
    plot_cov_ellipse(mu,cov,fill=None,color=colors[i])
plt.savefig("imgs/gmm_gibbs_ex.pdf")

#%%
pi = chain['mix.pi'][:,0] + 0
mus = np.concatenate([chain['theta.{}.A'.format(k)] for k in range(model.components)],-1)
mu = mus.copy()
for k in range(model.components):
    pi[:,k] = chain['mix.pi'][np.arange(pi.shape[0]),0,tau[:,k]]
    mu[:,:,k] = mus[np.arange(pi.shape[0]),:,tau[:,k]]

plt.figure()
plt.plot(pi)

#%%
modeli = InfiniteGMM(collapse_locally=True,sigma_ev=1)
sampleri = Gibbs()

#%%
sampleri.fit(data,modeli,samples=100)

#%%
sampleri.get_estimates(burn_rate=.9)

z_hat = modeli.z
scattercat(data.output,z_hat,figsize=figsize,colors=colors)
for k in np.unique(z_hat):
    idx = z_hat == k
    mu,S,nu = modeli._predictive_parameters(*modeli._posterior(modeli.y[idx],*modeli.theta))
    cov = S * (nu)/(nu-2)
    plot_cov_ellipse(mu,cov,fill=None,color=colors[k])
plt.savefig("imgs/gmm_dp_ex.pdf")

chain = sampleri.get_chain(burn_rate=0,flatten=False)
fig,ax = plt.subplots(len(chain),figsize=(5,1.5*len(chain)))
for ii,p in enumerate(chain):
    _x = chain[p]
    _x = _x.reshape(_x.shape[0],-1)
    ax[ii].plot(_x,'k',alpha=.1)
    ax[ii].set_title(p)
plt.tight_layout()
plt.savefig("imgs/gmm_dp_chain.pdf")

# %%