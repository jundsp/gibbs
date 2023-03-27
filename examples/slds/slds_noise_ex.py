#%%
from gibbs import Gibbs, SLDS, slds_test_data, tqdm, get_scatter_kwds, get_colors, Data, categorical2multinomial
import numpy as np
import matplotlib.pyplot as plt
import os
from sequential import slds, lds
from scipy.stats import invgamma

plt.style.use('sines-latex')



T = 200
M = 1
np.random.seed(123)

fvec = np.zeros(T)
f = np.random.uniform(0/T,.1)
for t in range(T):
    if t % 40 == 0:
        f = np.random.uniform(0/T,.1)
    fvec[t] = f

time = np.arange(T)
# fvec = np.sin(np.pi*time/T)*.1
phase = np.cumsum(2*np.pi*fvec)
y = np.cos(phase)
y = y[:,None]

y = np.concatenate([y]*M,0)
time = np.concatenate([time]*M,0)
y += np.random.normal(0,.5,y.shape)

data = Data(y=y,time=time)
data = data.filter((time < T//2-20) | (time > (T//2+10)))
data.plot()

#%%

states = 10
_C = np.zeros((1,2))
_C[0,0] = 1
ratio = 1
sigma = 1
_R = np.eye(1) * sigma ** 2
_Q = np.eye(2)* (sigma*ratio)**2
_m0 = np.zeros((2,1))
_P0 = np.eye(2)

model = SLDS(output_dim=1,state_dim=2,states=states,hyper_sample=False,expected_duration=50,learn_hmm=False,learn_lds=False,full_covariance=False,circular=True)
freqs = np.linspace(0,.1,model.states)
for k in range(model.states):
    model.lds.theta[k].sys._parameters['A'] = lds.rotation_matrix(2*np.pi*freqs[k])
    model.lds.theta[k].sys._parameters['Q'] = _Q
    model.lds.theta[k].obs._parameters['A'] = _C
    model.lds.theta[k].obs._parameters['Q'] = _R
    model.lds.theta[k].pri._parameters['A'] = _m0
    model.lds.theta[k].pri._parameters['Q'] = _P0



sampler = Gibbs()

#%%
_y = np.zeros((T,model.output_dim))
_x = np.zeros((T,model.state_dim))

#%%
for iter in tqdm(range(50)):
    model(data)
    sampler.step(model.named_parameters())

    for t in range(T):
        _y[t] = model.lds.C(model.hmm.z[t]) @ model.lds.x[t]

    y_tilde = data.output - _y[data.time]
    N = len(data)
    eps = np.trace(y_tilde.T @ y_tilde)

    a0 = 2
    b0 = a0 * (.1)**2
    a = a0 + 1/2*N
    b = b0 + 1/2*eps
    sigma2 = invgamma.rvs(a=a,scale=b)
    for m in model.lds.theta:
        m.obs._parameters['Q'] = np.eye(1)*sigma2

    for t in range(1,T):
        _x[t] = model.lds.A(model.hmm.z[t]) @ model.lds.x[t-1]

    x_tilde = model.lds.x[1:] - _x[1:]
    N = x_tilde.size
    eps = np.trace(x_tilde.T @ x_tilde)

    a0 = 1
    b0 = a0 * (.01)**2
    a = a0 + 1/2*N
    b = b0 + 1/2*eps
    sigma2 = invgamma.rvs(a=a,scale=b)
    for m in model.lds.theta:
        m.sys._parameters['Q'] = np.eye(model.state_dim)*sigma2

sampler.get_estimates()

#%%
chain = sampler.get_chain(burn_rate=0,flatten=False)
sampler.get_estimates(burn_rate=.9)
x_hat = sampler._estimates['lds.x']
z_hat = categorical2multinomial(chain['hmm.z']).mean(0).argmax(-1)
# x_hat = model.lds.x
# z_hat = model.hmm.z
y_hat = np.zeros((T,model.output_dim))
for t in range(z_hat.shape[0]):
    y_hat[t] = x_hat[t] @ sampler._estimates['lds.theta.{}.obs.A'.format(z_hat[t])].T

plt.plot(chain['hmm.z'].T,'k.',alpha=.1,markersize=2)

#%%
fig,ax = plt.subplots(2,figsize=(5,5))
colors = plt.get_cmap('jet')(np.linspace(0,1,states))
for k in range(y.shape[-1]):
    ax[0].scatter(data.time,data.output[:,k],c=colors[z_hat[data.time]],s=10,edgecolor='none')
ax[0].plot(y_hat,'k',alpha=.7)
ax[0].set_xlim(0,T)
ax[0].set_ylim(y.min()-.1,y.max()+.1)
ax[0].set_ylabel('$y$')
ax[1].scatter(np.arange(len(z_hat)),z_hat+1,c=colors[z_hat],s=20,edgecolor='none')
ax[1].set_xlabel('time (sample)'), ax[1].set_ylabel('state'),
ax[1].set_xlim(0,T), ax[1].set_ylim(1-.25,model.states+.25), ax[1].set_yticks(np.arange(model.states)+1)
plt.tight_layout()

path_out = "imgs"
os.makedirs(path_out,exist_ok=True)
plt.savefig(os.path.join(path_out,"slds_ex.pdf"))

# %%
plt.figure(figsize=(4,2))
plt.imshow(colors[chain['hmm.z']])
plt.savefig(os.path.join(path_out,"slds_ex_chain.pdf"))
plt.xlabel('time')
plt.ylabel('sample')

# %%
plt.plot(chain['lds.theta.0.obs.Q'][:,0,0]**.5)
plt.plot(chain['lds.theta.0.sys.Q'][:,0,0]**.5)
plt.ylim(0)

# %%
plt.plot(chain['lds.x'][:,:,0].T,'k',alpha=.05)

# %%
