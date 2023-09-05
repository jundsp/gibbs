#%%
from gibbs import Gibbs, GMM, gmm_generate, plot_cov_ellipse, get_colors, get_scatter_kwds,scattercat, Data, relabel, categorical2multinomial
import numpy as np
import os
import matplotlib.pyplot as plt

plt.style.use('gibbs.mplstyles.latex')

np.random.seed(123)
y = gmm_generate(500,2,5)[0]
data = Data(y=y)
data.plot()

#%%
model = GMM(components=10,hyper_sample=True)
sampler = Gibbs()

#%%
sampler.fit(data,model,samples=500)

#%%
chain = sampler.get_chain(burn_rate=.1)
tau = relabel(probs=chain['mix.rho'],verbose=True,iters=5)

pi = chain['mix.pi'][:,0]
pi = np.take_along_axis(pi,tau,-1)

rho = chain['mix.rho']
rho = np.take_along_axis(rho,tau[:,None,:],-1)
z_hat = rho.mean(0).argmax(-1)

mus = np.stack([chain['theta.{}.A'.format(k)][:,:,0] for k in range(model.components)],-1)
mu = np.take_along_axis(mus,tau[:,None,:],-1)
mu_hat = mu.mean(0)

Sigmas = np.stack([chain['theta.{}.Q'.format(k)] for k in range(model.components)],-1)
Sigma = np.take_along_axis(Sigmas,tau[:,None,None,:],-1)
Sigma_hat = Sigma.mean(0)

#%%
colors = get_colors()
scattercat(data.output,z_hat)
for k in np.unique(z_hat):
    plot_cov_ellipse(mu_hat[:,k],Sigma_hat[:,:,k],fill=None,color=colors[k])
plt.tight_layout()
# plt.savefig("imgs/gmm_data_orig.pdf")


plt.figure(figsize=(4,3))
for k in range(pi.shape[-1]):
    plt.plot(pi[:,k],color=colors[k],alpha=1,linewidth=.5)
plt.xlim(0,pi.shape[0]-1)
plt.ylim(0)
plt.xlabel('sample')
plt.ylabel(r"$\pi_k$")
plt.tight_layout()
# plt.savefig("imgs/gmm_pi_orig.pdf")

# %%
