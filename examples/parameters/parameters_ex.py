from gibbs import Gibbs
from gibbs.modules.parameters import NormalWishart
import numpy as np
import matplotlib.pyplot as plt


theta = NormalWishart(output_dim=2,input_dim=2,full_covariance=False,transform_sample=True)
sampler = Gibbs()

N = 100
y = np.random.randn(N,2)*(.1**.5) + 1
x = np.ones((N,2))
mask = np.ones(N).astype(bool)

for iter in range(100):
    theta.forward(y=y,x=x,mask=mask)
    sampler.step(theta.named_parameters())

chain = sampler.get_chain()

fig,ax = plt.subplots(len(chain))
for k,c in enumerate(chain):
    ax[k].plot(chain[c].reshape(chain[c].shape[0],-1))
    ax[k].set_title(c)
plt.tight_layout()
