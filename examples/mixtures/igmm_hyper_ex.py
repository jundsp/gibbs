#%%
import gibbs
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.style.use('gibbs.mplstyles.latex')

figsize=(3,2.5)
colors = gibbs.get_colors()
np.random.seed(123)
y, z = gibbs.gmm_generate(500,2,5)
y[z==0] = stats.norm.rvs(0,7,y[z==0].shape)
# y[z==1] = stats.norm.rvs(1,5,y[z==1].shape)
data = gibbs.Data(y=y)

plt.figure(figsize=figsize)
plt.scatter(data.output[:,0],data.output[:,1],color=colors[z],alpha=.5,edgecolors='none',s=15,linewidths=0)

#%%
model = gibbs.InfiniteGMM_Hyper(collapse_locally=True,sigma_ev=.5,sigma_outlier_ev=10)
sampler = gibbs.Gibbs()

#%%dcxsfv
sampler.fit(data,model,samples=20)

#%%
chain = sampler.get_chain(burn_rate=.9)
z_hat = gibbs.categorical2multinomial(chain['z']).mean(0).argmax(-1)

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
    theta_now = model.thetas[model.kinds[k]]
    mu,S,nu = model._predictive_parameters(*model._posterior(model.y[idx],*theta_now))
    cov = S * (nu)/(nu-2)
    gibbs.plot_cov_ellipse(mu,cov,fill=None,color=colors[k],linestyle=styles[model.kinds[k]])
plt.scatter(data.output[:,0],data.output[:,1],color=colors[z_hat],alpha=.5,edgecolors='none',s=15,linewidths=0,zorder=2)
# plt.xlim(-20,20)
# plt.ylim(-20,20)
plt.tight_layout()
plt.xlabel(r"$y_1$")
plt.ylabel(r"$y_2$")
plt.tight_layout()
plt.savefig("imgs/outlier_2d.pdf")

# %%
