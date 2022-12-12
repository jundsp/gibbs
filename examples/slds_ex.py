#%%
from gibbs import Gibbs, SLDS, slds_generate, tqdm, get_scatter_kwds, get_colors
import numpy as np
import matplotlib.pyplot as plt
import os

T = 200
np.random.seed(123)
y = slds_generate(T=T)
y = y[:,[0]]
y = y[:,None]
mask = np.ones(y.shape[:2]).astype(bool)
T = y.shape[0]
plt.plot(y[:,0])

#%%
np.random.seed(123)
model = SLDS(output_dim=1,state_dim=8,states=6,hyper_sample=True,expected_duration=100)
sampler = Gibbs()
logl = []
#%%
iters = 20
for iter in tqdm(range(iters)):
    model(y,mask=mask)
    sampler.step(model.named_parameters())
    logl.append(model.loglikelihood(y=y,mask=mask).sum())

plt.plot(logl)
plt.xlabel('sample'),plt.ylabel('log likelihood')
#%%
sampler.get_estimates()
x_hat = sampler._estimates['lds.x']
z_hat = sampler._estimates['hmm.z']
y_hat = x_hat @ sampler._estimates['lds.theta.0.obs.A'].T
Y_hat = np.zeros((T,model.states)) + np.nan
for t in range(z_hat.shape[0]):
    y_hat[t] = x_hat[t] @ sampler._estimates['lds.theta.{}.obs.A'.format(z_hat[t])].T
    Y_hat[t,z_hat[t]] = y_hat[t]
#%%
fig,ax = plt.subplots(2,figsize=(5,5))

rz,cz = np.nonzero(mask)
for k in range(y.shape[-1]):
    ax[0].scatter(rz,y[rz,cz,k],c='b',**get_scatter_kwds())
# ax[0].plot(y_chain.transpose(1,0,2)[:,:,0],'g',alpha=4/y_chain.shape[0]);
ax[0].plot(Y_hat)
# plt.plot(y_hat);
ax[0].set_xlim(0,T)
ax[0].set_ylim(y.min()-.5,y.max()+.5)
ax[0].set_ylabel('$y_{1,2}$')
ax[1].plot(z_hat,'k.')
ax[1].set_xlabel('time (sample)'), ax[1].set_ylabel('state'),
ax[1].set_xlim(0,T), ax[1].set_ylim(0,model.states-.5), ax[1].set_yticks(np.arange(model.states))
plt.tight_layout()

path_out = "imgs"
os.makedirs(path_out,exist_ok=True)
plt.savefig(os.path.join(path_out,"slds_ex.pdf"))
# %%

chain = sampler.get_chain(burn_rate=0,flatten=False)
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
# %%
