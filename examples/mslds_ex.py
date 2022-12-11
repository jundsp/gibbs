#%%
from gibbs import Gibbs, MSLDS, slds_generate, tqdm, get_scatter_kwds, get_colors, logsumexp
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
T = 200
np.random.seed(123)
y = slds_generate(T=T)
y = np.stack([y]*4,1)
y[:,:2] = np.sin(2*np.pi*np.arange(T)/200*.75)[:,None,None]
y += np.random.normal(0,1e-1,y.shape)

mask = np.ones(y.shape[:2]).astype(bool)
T = y.shape[0]
plt.plot(y[:,:,0])
#%%
np.random.seed(123)
model = MSLDS(output_dim=2,state_dim=2,components=2,states=4)
sampler = Gibbs()

logl = []
#%%
iters = 100
for iter in tqdm(range(iters)):
    model(y,mask=mask)
    sampler.step(model.named_parameters())
    logl.append(model.loglikelihood(y,mask=mask).sum((0,1)))

plt.plot(np.exp(np.stack(logl,0)))

sampler.get_estimates()

#%%
plt.figure(figsize=(4,3))
colors = get_colors()
rz,cz = np.nonzero(mask)
c_hat = sampler._estimates['mix.z']
plt.scatter(rz,y[rz,cz,0],c=colors[c_hat],**get_scatter_kwds())
# plt.plot(y_chain.transpose(1,0,2)[:,:,0],'g',alpha=1/y_chain.shape[0]);
# plt.plot(y_ev,'g')

for k in np.unique(c_hat):
    z_hat = sampler._estimates['slds.{}.hmm.z'.format(k)]
    x_hat = sampler._estimates['slds.{}.lds.x'.format(k)]
    y_hat = x_hat @ sampler._estimates['slds.{}.lds.theta.0.obs.A'.format(k)].T
    for t in range(y_hat.shape[0]):
        y_hat[t] = x_hat[t] @ sampler._estimates['slds.{}.lds.theta.{}.obs.A'.format(k,z_hat[t])].T
    plt.plot(y_hat,c=colors[k]);
plt.xlim(0,T)
plt.ylim(y.min()-.2,y.max()+.2)
plt.xlabel('time (sample)'), plt.ylabel('$y_1$')
plt.tight_layout()

path_out = "imgs"
os.makedirs(path_out,exist_ok=True)
# plt.savefig(os.path.join(path_out,"mslds_ex.pdf"))

# %%
logrho = model.loglikelihood(y,mask=mask)
logrho -= logsumexp(logrho,-1)[:,:,None]
logrho = logrho.reshape(-1,logrho.shape[-1])
plt.imshow(np.exp(logrho).T,aspect='auto',interpolation='none')
# %%
