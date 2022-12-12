from gibbs import Gibbs, gmm_generate, tqdm, plot_cov_ellipse, get_colors, get_scatter_kwds
from gibbs.modules.mixture import InfiniteMixture
import numpy as np
import os
import matplotlib.pyplot as plt


model = InfiniteMixture(learn=True)
sampler = Gibbs()

z = np.random.randint(0,3,(100))
for epoch in range(100):
    model(z)
    sampler.step(model.named_parameters())







chain = sampler.get_chain()
fig,ax = plt.subplots(len(chain),figsize=(5,1.5*len(chain)))
for ii,p in enumerate(chain):
    if ('.x' in p) | ('.z' in p):
        _x = chain[p]
        _x = np.swapaxes(_x,0,1)
        _x = _x.reshape(_x.shape[0],-1)
    else:
        _x = chain[p]
        _x = _x.reshape(_x.shape[0],-1)
    ax[ii].plot(_x,'k',alpha=.1)
    ax[ii].set_title(p)
plt.tight_layout()