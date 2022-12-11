#%%
from gibbs import Gibbs, LDS, lds_generate, tqdm, get_scatter_kwds, get_colors
from sequential.lds import polynomial_matrix
import numpy as np
import matplotlib.pyplot as plt
import os


#%%
T = 200
np.random.seed(123)
y = lds_generate(T=T)
y *= np.sin(2*np.pi*np.arange(T)/100*6)[:,None]
y = y[:,[0]]
y = np.stack([y]*2,1)
y += np.random.normal(0,2e-1,y.shape)
mask = np.ones(y.shape[:2]).astype(bool)
mask[70:130] = False
mask[:20] = False
mask[-20:] = False

#%%
np.random.seed(123)
model = LDS(output_dim=2,state_dim=8,parameter_sampling=True,hyper_sample=True)
# model.theta[0].sys._parameters['A'] = polynomial_matrix(3)
# model.theta[0].sys._parameters['Q'] = np.eye(3)*1e-2
# C = np.zeros((1,3))
# C[0,0] = 1
# model.theta[0].obs._parameters['A'] = C
# model.theta[0].obs._parameters['Q'] = np.eye(1)*1e-1
sampler = Gibbs()

logl = []
#%%
iters = 600
for iter in tqdm(range(iters)):
    model(y,mask=mask)
    sampler.step(model.named_parameters())
    logl.append(model.loglikelihood(y,mask=mask).sum())

plt.plot(logl)
#%%
sampler.get_estimates()
x_hat = sampler._estimates['x']
y_hat = x_hat @ sampler._estimates['theta.0.obs.A'].T

chain = sampler.get_chain(burn_rate=.8)
y_chain = chain['x'] @ chain['theta.0.obs.A'].transpose(0,2,1)
y_ev = y_chain.mean(0)
rz,cz = np.nonzero(mask)

#%%
plt.figure(figsize=(4,3))

plt.scatter(rz,y[rz,cz,0],c='b',**get_scatter_kwds())
plt.plot(y_chain.transpose(1,0,2)[:,:,0],'g',alpha=1/y_chain.shape[0]);
plt.plot(y_ev,'g')
# plt.plot(y_hat);
plt.xlim(0,T)
plt.ylim(y.min()-.5,y.max()+.5)
plt.xlabel('time (sample)'), plt.ylabel('$y_1$')
plt.tight_layout()

path_out = "imgs"
os.makedirs(path_out,exist_ok=True)
plt.savefig(os.path.join(path_out,"lds_fixed_ex.pdf"))

# %%
plt.plot(model.loglikelihood(y,mask=mask))

# %%
chain = sampler.get_chain(burn_rate=.8,flatten=False)
fig,ax = plt.subplots(len(chain),figsize=(5,1.5*len(chain)))
for ii,p in enumerate(chain):
    if p == 'x':
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
R_hat = sampler._estimates['theta.0.obs.Q']**.5
print(R_hat)

# %%
