#%%
from gibbs import Gibbs, lds_generate, tqdm, get_scatter_kwds, get_colors, LDS, Data
import numpy as np
import matplotlib.pyplot as plt
import os


#%%
T = 300
np.random.seed(123)
f = np.array([6.5/200, 8.25/200])
theta = 2*np.pi*np.random.randn(len(f))
n = np.arange(T)
x1 = np.sin(2*np.pi*n[:,None] * f[None,:] + theta[None,:]).sum(-1)
x2 = np.cos(2*np.pi*n[:,None] * f[None,:] + theta[None,:]).sum(-1)
x = np.stack([x1],-1)
x /= np.abs(x).max()
x += np.random.normal(0,.3,x.shape)
y = x[:,None,:]

mask = np.ones(y.shape[:2]).astype(bool)
mask[200:] = False

rz,cz = np.nonzero(mask)
time = n[rz]
y = y[rz,cz]

data = Data(y=y,time=time)
# data._T = T
data.plot()

#%%
np.random.seed(123)
model = LDS(output_dim=1,state_dim=8,parameter_sampling=True,full_covariance=True,hyper_sample=False)
sampler = Gibbs()

logl = []
#%%
# data._T = 400
sampler.fit(data=data,model=model,samples=100)

#%%
sampler.get_estimates()
x_hat = sampler._estimates['x']
y_hat = x_hat @ sampler._estimates['theta.0.obs.A'].T

chain = sampler.get_chain(burn_rate=0)
y_chain = chain['x'] @ chain['theta.0.obs.A'].transpose(0,2,1)
y_ev = y_chain.mean(0)
rz,cz = np.nonzero(mask)

#%%
plt.figure(figsize=(4,3))

plt.scatter(data.time,data.output,c='b',**get_scatter_kwds())
plt.plot(y_chain.transpose(1,0,2)[:,:,0],'r',alpha=4/y_chain.shape[0]);
plt.plot(y_ev,'b',linewidth=1)
# plt.plot(y_hat,'g');
plt.xlim(0,data.T)
plt.ylim(y.min()-.2,y.max()+.2)
plt.xlabel('time (sample)'), plt.ylabel('$y$')
plt.tight_layout()

path_out = "imgs"
os.makedirs(path_out,exist_ok=True)
plt.savefig(os.path.join(path_out,"lds_ex.pdf"))

# %%
plt.plot(model.loglikelihood(data))
# %%
chain = sampler.get_chain(burn_rate=0,flatten=False)
fig,ax = plt.subplots(len(chain),figsize=(5,1.5*len(chain)))
for ii,p in enumerate(chain):
    if ('x' in p) | ('.z' in p):
        _x = chain[p]
        _x = np.swapaxes(_x,0,1)
        _x = _x.reshape(_x.shape[0],-1)
    else:
        _x = chain[p]
        _x = _x.reshape(_x.shape[0],-1)
    ax[ii].plot(_x,'k',alpha=.1)
    ax[ii].set_title(p)
plt.tight_layout()
# %%
