#%%
from gibbs import Gibbs, SLDS, slds_generate, tqdm, get_scatter_kwds, get_colors, Data
import numpy as np
import matplotlib.pyplot as plt
import os
from sequential import slds

T = 200
np.random.seed(123)
y = slds_generate(T=T)
y = y[:,[0]]
y = y[:,None]
mask = np.ones(y.shape[:2]).astype(bool)
T = y.shape[0]
rz,cz = np.nonzero(mask)
n = np.arange(T)
time = n[rz]
y = y[rz,cz]

data = Data(y=y,time=time)
# data._T = T
data.plot()

#%%
model_vb = slds.SLDS(output_dim=1,state_dim=4,switch_dim=4,expected_duration=10,driven=False)

#%%
model_vb.train(y,epochs=10)
#%%
model_vb.plot()
#%%
#* Tighter prior over the covariance. 
model = SLDS(output_dim=1,state_dim=4,states=4,hyper_sample=False,expected_duration=10,learn_hmm=False,learn_lds=True,full_covariance=False)
sampler = Gibbs()

#%%
sampler.fit(data=data,model=model,samples=5)

#%%
chain = sampler.get_chain(burn_rate=.5,flatten=False)
sampler.get_estimates(burn_rate=.95)
x_hat = sampler._estimates['lds.x']
z_hat = sampler._estimates['hmm.z']
x_hat = model.lds.x
z_hat = model.hmm.z
y_hat = np.zeros((T,model.output_dim))
for t in range(z_hat.shape[0]):
    y_hat[t] = x_hat[t] @ sampler._estimates['lds.theta.{}.obs.A'.format(z_hat[t])].T

plt.plot(chain['hmm.z'].T,'k.',alpha=.1,markersize=2)

#%%
fig,ax = plt.subplots(2,figsize=(5,5))

rz,cz = np.nonzero(mask)
for k in range(y.shape[-1]):
    ax[0].scatter(data.time,data.output[:,k],c='b',**get_scatter_kwds())
ax[0].plot(y_hat)
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


fig,ax = plt.subplots(len(chain),figsize=(5,1.5*len(chain)))
for ii,p in enumerate(chain):
    if ('.x' in p) :
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


# %%
