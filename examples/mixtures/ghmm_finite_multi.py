#%%
from gibbs import Gibbs, tqdm, get_colors, Data, HMM, categorical2multinomial
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from itertools import product

T = 200
sources = 2
N = 10

sigma1 = 1
sigma2 = .05
mu1 = 0
mu2 = 0

y1 = np.random.normal(mu1,sigma1,(T,N))
y2 = np.random.normal(mu2,sigma2,(T,N))
y = y1.copy()
y[20:80] = y2[20:80]
y[-80:-20] = y2[-80:-20]
y[:,5:] = y1[:,5:]
y = y.ravel()
time = np.arange(T).repeat(N)

data = Data(y=y[:,None],time=time)
data = data.filter((data.time < 100) | (data.time > 105))
data = data.filter(np.abs(data.output[:,0]) < 1)
plt.plot(data.time,data.output,'.')

N_on = []
for t in range(data.T):
    N_on.append((data.time==t).sum())

#%%
logl = np.zeros((len(data),2))
logl[:,0] = norm.logpdf(data.output[:,0],mu1,sigma1)
logl[:,1] = norm.logpdf(data.output[:,0],mu2,sigma2)

#%%
Comb = product(np.arange(sources), repeat=N)
Comb = np.array(list(Comb))
Comb = np.concatenate([Comb,Comb[[0]]],0)
states = Comb.shape[0]

#%%
circular = False
states_on = states - 2
Gam = np.eye(states)
Gam[0,1] = 1e-1 / states_on
Gam[1:-1,1:-1] = 1 / states_on
Gam[1:-1,-1] = 1e-3 / states_on
Gam[-1,0] = 1e-3 * circular
# Gam[:] = 1
pi = np.zeros(states)
pi[0] = 1
pi[1:-1] = .5 / states_on
Gam /= Gam.sum(-1)[:,None]
pi /= pi.sum()

hmm = HMM(states=states,parameter_sampling=False)
hmm.set_parameters(Gamma=Gam,pi=pi)

# %%
logl_hmm = np.zeros((T,hmm.states)) - np.inf
for t in range(T):
    t_on = data.time == t
    _logl = logl[t_on]
    if N_on[t] > 0:
        Comb_ = Comb[:,:N_on[t]]
        idx_unique = np.unique(Comb_,axis=0,return_index=True)[-1]
        idx_unique = np.append(idx_unique,states-1)
        for k in idx_unique:
            logl_hmm[t,k] = _logl[np.arange(N_on[t]),Comb_[k]].sum()
    else:
        logl_hmm[t,[0,-1]] = 0
    
#%%
sampler = Gibbs()
#%%
for sample in tqdm(range(10)):    
    hmm.forward(logl_hmm)
    sampler.step(hmm.named_parameters())
sampler.get_estimates()

# %%

chain = sampler.get_chain()
z_hat = categorical2multinomial(chain['z']).mean(0).argmax(-1)

#%%
colors = get_colors()
plt.imshow(chain['z'])

OnOff = np.zeros(hmm.states).astype(int)
OnOff[0] = 0
OnOff[1:-1] = 1
OnOff[-1] = 2

fig, ax = plt.subplots(2,figsize=(4,3),sharex=True,gridspec_kw={'height_ratios': [1, 10]})
for t in range(data.T):
    t_on = data.time == t
    ax[1].scatter(data.time[t_on],data.output[t_on],c=colors[Comb[z_hat[t],:N_on[t]].ravel()],edgecolor='none',s=10,alpha=.9)
ax[1].set_xlim(0,data.T-1)
ax[0].imshow(colors[OnOff[z_hat]][None,:,:])
ax[0].set_yticks([])
plt.tight_layout()

# %%
