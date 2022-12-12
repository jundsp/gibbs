#%%
from gibbs import Gibbs, hmm_generate, tqdm, get_colors, get_scatter_kwds, plot_cov_ellipse
from gibbs.modules.hmm import GHMM
import numpy as np
import matplotlib.pyplot as plt
import os

N = 500
np.random.seed(123)
y,z_true = hmm_generate(n=N,n_components=3)
y = np.stack([y]*1,1)
# y += np.random.normal(0,1e-1,y.shape)
mask = np.ones(y.shape[:2]).astype(bool)
mask[200:300] = False
#%%
np.random.seed(123)
model = GHMM(output_dim=2,states=6)
sampler = Gibbs()

#%%
iters = 100
for iter in tqdm(range(iters)):
    model(y,mask=mask)
    sampler.step(model.named_parameters())

# %%
sampler.get_estimates()

# %%
z_hat = sampler._estimates['hmm.z']

# %%
plt.figure(figsize=(4,3))
colors = get_colors()
kwds = get_scatter_kwds()
rz, cz = np.nonzero(mask)
plt.scatter(y[rz,cz,0],y[rz,cz,1],c=colors[z_hat[rz]],**kwds)
for k in np.unique(z_hat):
    mu = sampler._estimates['theta.{}.A'.format(k)].ravel()
    Sigma = sampler._estimates['theta.{}.Q'.format(k)]
    plot_cov_ellipse(mu,Sigma,nstd=1,fill=None,color=colors[k],linewidth=2)
plt.xlabel('$y_1$'), plt.ylabel('$y_2$')
plt.tight_layout()

path_out = "imgs"
os.makedirs(path_out,exist_ok=True)
plt.savefig(os.path.join(path_out,"ghmm_ex.pdf"))

# %%
fig,ax = plt.subplots(2)
ax[0].imshow(np.atleast_2d(z_true),aspect='auto')
ax[1].imshow(np.atleast_2d(z_hat),aspect='auto')

# %%
