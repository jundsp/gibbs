#%%
from gibbs import Gibbs, SLDS, slds_generate, tqdm, get_scatter_kwds, get_colors
import numpy as np
import matplotlib.pyplot as plt
import os


#%%
T = 200
np.random.seed(123)
y = slds_generate(T=T)
# y += np.sin(2*np.pi*np.arange(T)/T*2.5)[:,None]
y = np.stack([y]*1,1)
y += np.random.normal(0,1e-1,y.shape)
mask = np.ones(y.shape[:2]).astype(bool)
# mask[70:130] = False

#%%
np.random.seed(123)
model = SLDS(output_dim=2,state_dim=8,states=1,parameter_sampling=True)
sampler = Gibbs()

#%%
iters = 100
for iter in tqdm(range(iters)):
    model(y,mask=mask)
    sampler.step(model.named_parameters())

#%%
sampler.get_estimates()
x_hat = sampler._estimates['lds.x']
z_hat = sampler._estimates['hmm.z']
y_hat = x_hat @ sampler._estimates['lds.theta.0.obs.A'].T

chain = sampler.get_chain(burn_rate=.5)
y_chain = chain['lds.x'] @ chain['lds.theta.0.obs.A'].transpose(0,2,1)
y_ev = y_chain.mean(0)
rz,cz = np.nonzero(mask)

#%%
plt.figure(figsize=(4,3))

plt.scatter(rz,y[rz,cz,0],c='b',**get_scatter_kwds())
plt.plot(y_chain.transpose(1,0,2)[:,:,0],'g',alpha=4/y_chain.shape[0]);
plt.plot(y_ev,'g')
# plt.plot(y_hat);
plt.xlim(0,T)
plt.ylim(y.min()-.5,y.max()+.5)
plt.xlabel('time (sample)'), plt.ylabel('$y_1$')
plt.tight_layout()

path_out = "imgs"
os.makedirs(path_out,exist_ok=True)
plt.savefig(os.path.join(path_out,"slds_ex.pdf"))

# %%
plt.plot(z_hat)
# %%
