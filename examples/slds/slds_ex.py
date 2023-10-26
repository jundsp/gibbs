#%%
import gibbs
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('gibbs.mplstyles.latex')

T = 100
M = 2
np.random.seed(123)
fvec = np.zeros(T)
f = np.random.uniform(0/T,.1)
for t in range(T):
    if t % 40 == 0:
        f = np.random.uniform(0/T,.1)
    fvec[t] = f

time = np.arange(T)
fvec = np.sin(np.pi*time/T)*.1
phase = np.cumsum(2*np.pi*fvec)
y = np.cos(phase)
y = y[:,None]

y = np.concatenate([y]*M,0)
time = np.concatenate([time]*M,0)
y += np.random.normal(0,.1,y.shape)

data = gibbs.Data(y=y,time=time)
# data._T = T
mask = np.ones(len(data)).astype(bool)
mask[data.time < 10] = False
mask[data.time > (data.T-10)] = False
data = data.filter(mask)
data.plot()

#%%
states = 10
_C = np.zeros((1,2))
_C[0,0] = 1
ratio = 1
sigma = .1
_R = np.eye(1) * sigma ** 2
_Q = np.eye(2)* (sigma*ratio)**2
_m0 = np.zeros((2,1))
_P0 = np.eye(2)

np.random.seed(1)

model = gibbs.SLDS(output_dim=1,state_dim=2,states=states,hyper_sample=False,expected_duration=10,learn_hmm=False,learn_lds=False,full_covariance=False,circular=True,hybrid=False)
freqs = np.linspace(0,.1,model.states)
for k in range(model.states):
    c,s = np.cos(2*np.pi*freqs[k]), np.sin(2*np.pi*freqs[k])
    model.lds.theta[k].sys._parameters['A'] = np.array([c,-s,s,c]).reshape(2,-1)
    model.lds.theta[k].sys._parameters['Q'] = _Q
    model.lds.theta[k].obs._parameters['A'] = _C
    model.lds.theta[k].obs._parameters['Q'] = _R
    model.lds.theta[k].pri._parameters['A'] = _m0
    model.lds.theta[k].pri._parameters['Q'] = _P0


sampler = gibbs.Gibbs()

#%%
sampler.fit(data=data,model=model,samples=200)

#%%
chain = sampler.get_chain(burn_rate=0,flatten=False)
sampler.get_estimates(burn_rate=.9)
x_hat = sampler._estimates['lds.x']
z_hat = sampler._estimates['hmm.z']

# x_hat = model.lds.x
# z_hat = model.hmm.z
steps = chain['lds.x'].shape[0]
y_chain = np.zeros((T,steps,model.output_dim))
for t in range(z_hat.shape[0]):
    y_chain[t] = (chain['lds.theta.{}.obs.A'.format(z_hat[t])] @ chain['lds.x'][:,t][...,None])[...,0]

y_hat = y_chain.mean(1)
y_chain = y_chain.reshape(y_chain.shape[0],-1)

plt.plot(chain['hmm.z'].T,'k.',alpha=.1,markersize=2)

#%%
fig,ax = plt.subplots(2,figsize=(4,4))
colors = plt.get_cmap('jet')(np.linspace(0,1,states))
for k in range(y.shape[-1]):
    ax[0].scatter(data.time,data.output[:,k],c=colors[z_hat[data.time]],s=10,edgecolor='none',zorder=2,alpha=.8)
ax[0].plot(y_chain,'k',alpha=.05,zorder=1)
ax[0].plot(y_hat,'k',alpha=.75,zorder=1)
ax[0].set_xlim(0,T)
ax[0].set_ylim(y.min()-.1,y.max()+.1)
ax[0].set_ylabel('$y$')
ax[1].scatter(np.arange(len(z_hat)),z_hat+1,c=colors[z_hat],s=20,edgecolor='none')
ax[1].set_xlabel('time'), ax[1].set_ylabel('state'),
ax[1].set_xlim(0,T), ax[1].set_ylim(1-.25,model.states+.25), ax[1].set_yticks(np.arange(model.states)+1)
plt.tight_layout()

path_out = "imgs"
os.makedirs(path_out,exist_ok=True)
plt.savefig(os.path.join(path_out,"slds_ex.png"))

# %%
plt.figure(figsize=(3.5,2))
plt.imshow(colors[chain['hmm.z']])
plt.xlabel('time')
plt.ylabel('sample')
plt.tight_layout()
plt.savefig(os.path.join(path_out,"slds_ex_chain.png"),bbox_inches="tight")

# %%
