#%%
from gibbs import Gibbs, get_colors, get_scatter_kwds, plot_cov_ellipse, Module, Data, HMM, tqdm,scattercat, mvn
import numpy as np
import matplotlib.pyplot as plt
import os
import hammer


class Harmonic(Module):
    def __init__(self,H=30,tau=9):
        super(Harmonic,self).__init__()
        self.H = H
        self.tau = tau

        self.var_harmonic = 1
        self.var_residual = 500

        self.initialize()

    def initialize(self):
        hammer_model = hammer.Harmonic(H=self.H,tau=self.tau)
        hammer_model.make_transition()
        Gamma,pi = hammer_model.transition,hammer_model.initial
        self.coeffs = hammer_model.k
        self.states = hammer_model.S

        self.hmm = HMM(states=self.states,parameter_sampling=False)
        self.hmm._parameters['Gamma'] = Gamma
        self.hmm._parameters['pi'] = pi
        self._parameters['x'] = np.ones(1)*50.0

    def sample_x(self,data,z):
        h_idx = z < self.H
        y = data.output[h_idx]
        k = self.coeffs[z[h_idx]]

        # Have var be sampled, or prior? 
        lam0 = (1e-2)**2.0
        m0 = 500.0
        ell0 = m0*lam0

        lam = (k.T @ k) / self.var_harmonic + lam0
        ell = (k.T @ y) / self.var_harmonic + ell0
        mu = ell / lam
        sigma = 1/lam
        self._parameters['x'] = mvn.rvs(mu,sigma).ravel()


    def loglikelihood(self,data):

        logl = np.zeros((len(data),self.states))
        xmu = data.output - self.coeffs[None,:] * self.x
        logl[:,:self.H] = -0.5*(np.log(2*np.pi*self.var_harmonic) + xmu**2.0 / self.var_harmonic)
        logl[:,self.H:] = -0.5*(np.log(2*np.pi*self.var_residual))
        return logl

    def forward(self,data:'Data'):
        logl = self.loglikelihood(data=data)
        self.hmm(logl)
        self.sample_x(data,z=self.hmm.z)


np.random.seed(123)
f0 = 440.1
N = 30
y = 1.0*np.arange(1,N+1)*f0
y += np.random.normal(0,2,y.shape)
noise = np.random.uniform(0,y.max(),N)
y[N//4:] = noise[N//4:]
y = np.sort(y)
y = y[:,None]

t = np.arange(y.shape[0])
features = dict(frequency=y,amplitude=y*0,nbd=y*0,ndd=y*0)
data = Data(y,time=t)

data.plot()

#%%
model = Harmonic()
sampler = Gibbs()

#%%
for iter in tqdm(range(1000)):
    model(data)
    sampler.step(model.named_parameters())

sampler.get_estimates(burn_rate=.75)
z_hat = sampler._estimates['hmm.z']
x_hat = sampler._estimates['x']
plt.plot(z_hat)
#%%
chain = sampler.get_chain(burn_rate=0)
plt.plot(chain['hmm.z'].T,'k',alpha=.2);

# %%
colors = get_colors()
labels = (z_hat>=model.H).astype(int)
plt.scatter(data.time*0+labels,data.output,c=colors[labels],s=10)
plt.ylim(0)
plt.grid()

# %%
plt.plot(chain['x'])
# %%
