#%%
import gibbs
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.style.use('gibbs.mplstyles.latex')

figsize=(3,2.5)
colors = gibbs.get_colors()
np.random.seed(123)
y, z = gibbs.gmm_generate(200,2,4)
y[z==0] = stats.laplace.rvs(0,3,y[z==0].shape)
data = gibbs.Data(y=y)

plt.figure(figsize=figsize)
plt.scatter(data.output[:,0],data.output[:,1],color=colors[z],alpha=.5,edgecolors='none',s=15,linewidths=0)

#%%
model = gibbs.InfiniteDistributionMix(sigma_ev=1,alpha=1,learn=True)
sampler = gibbs.Gibbs()

#%%
sampler.fit(data,model,samples=10)

#%%
chain = sampler.get_chain(burn_rate=0)
burn_rate = .9
start_sample = int(burn_rate * len(sampler))
z_hat = gibbs.categorical2multinomial(chain['z'][start_sample:]).mean(0).argmax(-1)

fig,ax = plt.subplots(len(chain),figsize=(5,1.5*len(chain)))
for ii,p in enumerate(chain):
    _x = chain[p]
    _x = _x.reshape(_x.shape[0],-1)
    ax[ii].plot(_x,'k',alpha=.1)
    ax[ii].set_title(p)
plt.tight_layout()

styles = ['-','--']

plt.figure(figsize=figsize)
for k in np.unique(z_hat):
    idx = z_hat == k
    _dist = model.dists[model.kinds[k]]
    mu, cov = _dist.params2moments(*_dist.predictive_parameters(*_dist.posterior(model.data.output[idx])))
    gibbs.plot_cov_ellipse(mu,cov,fill=None,color=colors[k],linestyle=styles[model.kinds[k]])
plt.scatter(data.output[:,0],data.output[:,1],color=colors[z_hat],alpha=.5,edgecolors='none',s=15,linewidths=0,zorder=2)
plt.tight_layout()


# %%
model = gibbs.FiniteDistributionMix(sigma_ev=1,alpha=1,learn=False,components=4)
sampler = gibbs.Gibbs()

#%%
sampler.fit(data,model,samples=30)

#%%
chain = sampler.get_chain(burn_rate=0)
burn_rate = .9
start_sample = int(burn_rate * len(sampler))
z_hat = gibbs.categorical2multinomial(chain['z'][start_sample:]).mean(0).argmax(-1)

fig,ax = plt.subplots(len(chain),figsize=(5,1.5*len(chain)))
for ii,p in enumerate(chain):
    _x = chain[p]
    _x = _x.reshape(_x.shape[0],-1)
    ax[ii].plot(_x,'k',alpha=.1)
    ax[ii].set_title(p)
plt.tight_layout()

styles = ['-','--']
plt.figure(figsize=figsize)
for k in np.unique(z_hat):
    idx = z_hat == k
    _dist = model.dists[model.kinds[k]]
    mu, cov = _dist.params2moments(*_dist.predictive_parameters(*_dist.posterior(model.data.output[idx])))
    gibbs.plot_cov_ellipse(mu,cov,fill=None,color=colors[k],linestyle=styles[model.kinds[k]])
plt.scatter(data.output[:,0],data.output[:,1],color=colors[z_hat],alpha=.5,edgecolors='none',s=15,linewidths=0,zorder=2)
plt.tight_layout()

# %%
