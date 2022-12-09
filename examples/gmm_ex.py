#%%
from gibbs import Gibbs, GMM, gmm_generate, tqdm
import numpy as np

N = 500
np.random.seed(123)
y = gmm_generate(n=N,n_components=3)[0]

#%%
np.random.seed(123)
model = GMM(output_dim=2,components=6)
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
