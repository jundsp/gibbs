#%%
from gibbs import Gibbs, LDS, lds_generate, tqdm
import numpy as np
import matplotlib.pyplot as plt

T = 100
np.random.seed(123)
y = lds_generate(T=T)

#%%
np.random.seed(123)
model = LDS(output_dim=2,state_dim=6)
# Have sampler point to parameters, retrieve new values somehow.
sampler = Gibbs()

#%%
iters = 100
for iter in tqdm(range(iters)):
    model(y)
    sampler.step(model.named_parameters())


#%%
sampler.get_estimates()
x_hat = sampler._estimates['x']
y_hat = x_hat @ sampler._estimates['theta.0.obs.A'].T
plt.plot(y_hat);
# %%

