#%%
from gibbs import Gibbs, get_colors, LDS, Data, Mixture, Plate, Module, categorical2multinomial
import numpy as np
import matplotlib.pyplot as plt

class MixtureLDS(Module):
    def __init__(self,output_dim,state_dim=2,components=2,learn=False):
        super().__init__()

        self.components = components

        param_sampling = [False] + [True]*(components-1)

        self.lds_plate = Plate(*[LDS(output_dim=output_dim,state_dim=state_dim,parameter_sampling=param_sampling[i],full_covariance=False,hyper_sample=False,init_method='predict') for i in range(components)])
        self.mix = Mixture(components=components,learn=True)

    def forward(self,data:'Data'):
        logl = np.zeros((len(data),self.components))
        if self.mix.z.shape[0] != len(data):            
            self.mix.forward(logl=logl)

        for k, lds in enumerate(self.lds_plate):
            _data = data.filter(self.mix.z==k)
            lds.forward(_data)
            logl[:,k] = lds.loglikelihood(data)
        self.mix.forward(logl=logl)

#%%
T = 50
np.random.seed(123)
time = np.arange(T)

f = .11

y1 = np.cos(time*f)
y2 = np.sin(time*f)
noise = [np.random.uniform(-1,1,y1.shape) for k in range(2)]
y = np.stack([y1,y1,*noise],0)
time = np.stack([time]*y.shape[0],0).ravel()
y = y.ravel()
y += np.random.normal(0,1e-2,y.shape)

data = Data(y=y[:,None],time=time)
data.plot()

#%%
np.random.seed(123)
model = MixtureLDS(components=3,output_dim=1,state_dim=2,learn=True)
sampler = Gibbs()

#%%
sampler.fit(data=data,model=model,samples=50)

# %%
chain = sampler.get_chain()
z_hat = categorical2multinomial(chain['mix.z']).mean(0).argmax(-1)
y_hat = np.stack([sampler._estimates['lds_plate.{}.x'.format(k)] @ sampler._estimates['lds_plate.{}.theta.0.obs.A'.format(k)].T for k in range(model.components)],-1)

colors = get_colors()
plt.figure(figsize=(3,2))
plt.imshow(colors[chain['mix.z']])
plt.xlabel('data point')
plt.ylabel('sample')
plt.tight_layout()
# plt.savefig('imgs/ldsmm_zchain_ex.pdf')

# %%
plt.figure(figsize=(3,2))
plt.scatter(data.time,data.output,color=colors[z_hat],linewidth=0,s=10)
for k in range(y_hat.shape[-1]):
    plt.plot(y_hat[:,0,k],color=colors[k])

plt.xlabel('time')
plt.tight_layout()
# plt.savefig('imgs/ldsmm_ex.pdf')

# %%
