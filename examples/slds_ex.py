#%%
from gibbs import Gibbs, SLDS, slds_generate, tqdm
import numpy as np
import matplotlib.pyplot as plt

T = 200
np.random.seed(123)
y = slds_generate(T=T)
T = y.shape[0]
plt.plot(y)

#%%
np.random.seed(123)
model = SLDS(output_dim=2,state_dim=2,states=4)
# Have sampler point to parameters, retrieve new values somehow.
sampler = Gibbs()

#%%
iters = 50
for iter in tqdm(range(iters)):
    model(y)
    sampler.step(model.named_parameters())

#%%
sampler.get_estimates(burn_rate=.8)
chain = sampler.get_chain(flatten=False,burn_rate=.8)

# %%
z_hat = sampler._estimates['hmm.z']
clen = len(chain['lds.x'])
y_samp = np.zeros((T,clen,model.output_dim,))
for s in range(clen):
    for t in range(T):
        y_samp[t,s] = chain['lds.x'][s][t] @ chain['lds.theta.{}.obs.A'.format(chain['hmm.z'][s][t])][s].T
y_hat = y_samp.mean(1)

#%%
fig, ax = plt.subplots(2,figsize=(6,5))
colors = np.array(['r','b','g','o','y','m','k'])
for k in range(y_samp.shape[-1]):
    ax[0].plot(y_samp[:,:,k],c=colors[k],alpha=1/clen);
ax[0].plot(y,'k.',alpha=.5)
for k in range(y_samp.shape[-1]):
    ax[0].plot(y_hat[:,k],linewidth=2,c=colors[k])
ax[0].set_ylabel('$y_t$')
ax[0].set_xlim(-.5,T)
ax[0].set_ylim(y.min()-.5,y.max()+.5)


ax[1].plot(z_hat,'k.')
ax[1].set_yticks(np.arange(model.states))
ax[1].set_xlabel('time (samples)')
ax[1].set_ylabel('state')
ax[1].set_xlim(-.5,T)
plt.tight_layout()
plt.savefig('gibbs_slds_test.pdf')
# %%
