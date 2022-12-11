#%%
from gibbs import Gibbs, MLDS, lds_generate, tqdm, get_scatter_kwds, get_colors, logsumexp
from sequential.lds import polynomial_matrix,polynomial_initial_covariance
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
T = 200
np.random.seed(123)
y = lds_generate(T=T)
y = np.sin(2*np.pi*np.arange(T)/200*1)[:,None]
y = y[:,[0]]
y = np.stack([y]*2,1)
y[:,:1] = np.cos(2*np.pi*np.arange(T)/200*3)[:,None,None]
y += np.random.normal(0,1e-1,y.shape)
mask = np.ones(y.shape[:2]).astype(bool)
# mask[70:130] = False
# mask[:20] = False
# mask[-20:] = False
plt.plot(y[:,:,0],'k.')
#%%
np.random.seed(123)
model = MLDS(output_dim=1,state_dim=2,components=3,parameter_sampling=True,hyper_sample=False)

for l in model.lds:
    l.theta[0].sys._parameters['A'] = polynomial_matrix(2)
    l.theta[0].sys._parameters['Q'] = np.diag([1e-3,1e-9])
    C = np.zeros((1,2))
    C[0,0] = 1
    l.theta[0].obs._parameters['A'] = C
    l.theta[0].obs._parameters['Q'] = np.eye(1)*1e-1

    l.theta[0].pri._parameters['A'] = np.zeros((2,1))
    l.theta[0].pri._parameters['Q'] = polynomial_initial_covariance(2)


sampler = Gibbs()

logl = []
#%%
iters = 50
for iter in tqdm(range(iters)):
    model(y,mask=mask)
    sampler.step(model.named_parameters())
    logl.append(model.loglikelihood(y,mask=mask).sum((0,1)))

plt.plot(np.stack(logl,0))

sampler.get_estimates()

#%%
plt.figure(figsize=(4,3))
colors = get_colors()
rz,cz = np.nonzero(mask)
plt.scatter(rz,y[rz,cz,0],c=colors[model.mix.z],**get_scatter_kwds())
# plt.plot(y_chain.transpose(1,0,2)[:,:,0],'g',alpha=1/y_chain.shape[0]);
# plt.plot(y_ev,'g')
z_hat = sampler._estimates['mix.z']
for k in np.unique(z_hat):
    x_hat = sampler._estimates['lds.{}.x'.format(k)]
    y_hat = x_hat @ sampler._estimates['lds.{}.theta.0.obs.A'.format(k)].T
    plt.plot(y_hat,c=colors[k]);
plt.xlim(0,T)
plt.ylim(y.min()-.2,y.max()+.2)
plt.xlabel('time (sample)'), plt.ylabel('$y_1$')
plt.tight_layout()

path_out = "imgs"
os.makedirs(path_out,exist_ok=True)
# plt.savefig(os.path.join(path_out,"mlds_ex.pdf"))

# %%
logrho = model.loglikelihood(y,mask=mask)
logrho -= logsumexp(logrho,-1)[:,:,None]
logrho = logrho.reshape(-1,logrho.shape[-1])
plt.imshow(np.exp(logrho).T,aspect='auto',interpolation='none')
# %%
