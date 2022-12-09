#%%
from gibbs import Gibbs, SLDS, slds_generate, tqdm, get_scatter_kwds, get_colors
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
T = 200
np.random.seed(123)
y = slds_generate(T=T)
# y = y[:,[0]]
# y += np.sin(2*np.pi*np.arange(T)/T*2.5)[:,None]
y = np.stack([y]*2,1)
y += np.random.normal(0,1e-1,y.shape)
mask = np.ones(y.shape[:2]).astype(bool)
mask[70:130] = False

#%%
np.random.seed(123)
model = SLDS(output_dim=2,state_dim=4,states=6,parameter_sampling=True)
sampler = Gibbs()

#%%
iters = 100
for iter in tqdm(range(iters)):
    model(y,mask=mask)
    sampler.step(model.named_parameters())

#%%
sampler.get_estimates()
x_hat = model.lds.x
z_hat = model.hmm.z
y_hat = x_hat @ sampler._estimates['lds.theta.0.obs.A'].T
for t in range(z_hat.shape[0]):
    y_hat[t] = x_hat[t] @ model.lds.theta[z_hat[t]].obs.A.T

chain = sampler.get_chain(burn_rate=.5)
y_chain = chain['lds.x'] @ chain['lds.theta.1.obs.A'].transpose(0,2,1)
y_ev = y_chain.mean(0)
rz,cz = np.nonzero(mask)

#%%
fig,ax = plt.subplots(2,figsize=(5,5))

for k in range(y.shape[-1]):
    ax[0].scatter(rz,y[rz,cz,k],c='b',**get_scatter_kwds())
ax[0].plot(y_chain.transpose(1,0,2)[:,:,0],'g',alpha=4/y_chain.shape[0]);
ax[0].plot(y_hat,'g')
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
