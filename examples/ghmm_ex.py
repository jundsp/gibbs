#%%
from gibbs import Gibbs, GHMM, hmm_generate, tqdm
import numpy as np
import matplotlib.pyplot as plt

N = 200
np.random.seed(123)
y,z_true = hmm_generate(n=N,n_components=3)

#%%
np.random.seed(123)
model = GHMM(output_dim=2,states=3)
# Have sampler point to parameters, retrieve new values somehow.
sampler = Gibbs()

#%%
iters = 100
for iter in tqdm(range(iters)):
    model(y)
    sampler.step(model.named_parameters())

# %%
sampler.get_estimates()

# %%
plt.plot(z_true)
plt.plot(sampler._estimates['hmm.z'],'--')
# %%
