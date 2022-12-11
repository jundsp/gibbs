#%%
from gibbs import Gibbs, GMM, gmm_generate, tqdm, plot_cov_ellipse, get_colors, get_scatter_kwds
import numpy as np
import os
import matplotlib.pyplot as plt

#%%
N = 200
np.random.seed(123)
y = gmm_generate(n=N,n_components=4)[0]
y = y[None,:,:]
#%%
np.random.seed(123)
model = GMM(output_dim=1,components=6)
sampler = Gibbs()

#%%
iters = 100
for iter in tqdm(range(iters)):
    model(y)
    sampler.step(model.named_parameters())

sampler.get_estimates()

z_hat = sampler._estimates['mix.z']
# %%

plt.figure(figsize=(4,3))
colors = get_colors()
kwds = get_scatter_kwds()
plt.scatter(y[0,:,0],y[0,:,1],c=colors[z_hat],**kwds)
for k in np.unique(z_hat):
    mu = sampler._estimates['theta.{}.A'.format(k)].ravel()
    Sigma = sampler._estimates['theta.{}.Q'.format(k)]
    plot_cov_ellipse(mu,Sigma,nstd=1,fill=None,color=colors[k],linewidth=2)
plt.xlabel('$y_1$'), plt.ylabel('$y_2$')
plt.tight_layout()

path_out = "imgs"
os.makedirs(path_out,exist_ok=True)
plt.savefig(os.path.join(path_out,"gmm_ex.pdf"))
# %%

