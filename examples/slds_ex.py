#%%
from gibbs import Gibbs, SLDS, slds_generate, tqdm, get_scatter_kwds, get_colors, Data
import numpy as np
import matplotlib.pyplot as plt
import os
from sequential import slds, lds

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
y += np.random.normal(0,5e-2,y.shape)

data = Data(y=y,time=time)
# data._T = T
data.plot()

#%%
model_vb = slds.SLDS(output_dim=1,state_dim=4,switch_dim=4,expected_duration=10,driven=False)

#%%
model_vb.train(y,epochs=1)
#%%
model_vb.plot()
#%%
#* Tighter prior over the covariance. 
C = np.zeros((1,2))
C[0,0] = 1
R = np.eye(1)*1e-1
model = SLDS(output_dim=1,state_dim=2,states=8,hyper_sample=False,expected_duration=20,learn_hmm=True,learn_lds=False,full_covariance=True)
for k in range(model.states):
    model.lds.theta[k].sys._parameters['A'] = lds.rotation_matrix(2*np.pi*k/100)
    model.lds.theta[k].sys._parameters['Q'] = np.eye(2)*1e-2
    model.lds.theta[k].obs._parameters['A'] = C
    model.lds.theta[k].obs._parameters['Q'] = R
    model.lds.theta[k].pri._parameters['A'] = np.zeros((2,1))
    model.lds.theta[k].pri._parameters['Q'] = np.eye(2)

sampler = Gibbs()

#%%
sampler.fit(data=data,model=model,samples=100)

#%%
chain = sampler.get_chain(burn_rate=0,flatten=False)
sampler.get_estimates(burn_rate=.75)
x_hat = sampler._estimates['lds.x']
z_hat = sampler._estimates['hmm.z']
# x_hat = model.lds.x
# z_hat = model.hmm.z
y_hat = np.zeros((T,model.output_dim))
for t in range(z_hat.shape[0]):
    y_hat[t] = x_hat[t] @ sampler._estimates['lds.theta.{}.obs.A'.format(z_hat[t])].T

plt.plot(chain['hmm.z'].T,'k.',alpha=.1,markersize=2)

#%%
fig,ax = plt.subplots(2,figsize=(5,5))
colors = get_colors()
rz,cz = np.nonzero(mask)
for k in range(y.shape[-1]):
    ax[0].scatter(data.time,data.output[:,k],c=colors[z_hat],s=20,edgecolor='none')
ax[0].plot(y_hat,'k',alpha=.7)
ax[0].set_xlim(0,T)
ax[0].set_ylim(y.min()-.5,y.max()+.5)
ax[0].set_ylabel('$y$')
ax[1].scatter(np.arange(len(z_hat)),z_hat,c=colors[z_hat],s=20,edgecolor='none')
ax[1].set_xlabel('time (sample)'), ax[1].set_ylabel('state'),
ax[1].set_xlim(0,T), ax[1].set_ylim(0,model.states-.5), ax[1].set_yticks(np.arange(model.states))
plt.tight_layout()

path_out = "imgs"
os.makedirs(path_out,exist_ok=True)
plt.savefig(os.path.join(path_out,"slds_ex.pdf"))
# %%
# fig,ax = plt.subplots(len(chain),figsize=(5,1.5*len(chain)))
# for ii,p in enumerate(chain):
#     if ('.x' in p) :
#         _x = chain[p]
#         _x = np.swapaxes(_x,0,1)
#         _x = _x.reshape(_x.shape[0],-1)
#     else:
#         _x = chain[p]
#         _x = _x.reshape(_x.shape[0],-1)
#     ax[ii].plot(_x,'k',alpha=.1)
#     ax[ii].set_title(p)
# plt.tight_layout()

# %%
