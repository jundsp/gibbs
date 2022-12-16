#%%
from gibbs import Gibbs, get_colors, get_scatter_kwds, plot_cov_ellipse, Module, Data, HMM, tqdm,scattercat, mvn, gamma
import numpy as np
import matplotlib.pyplot as plt
import os
import hammer
import sines

class Harmonic(Module):
    def __init__(self,H=30,tau=9):
        super(Harmonic,self).__init__()
        self.H = H
        self.tau = tau
        self.var_residual = 1000

        self.lam0 = (.2e-2)**2.0
        m0 = 500.0
        self.ell0 = m0*self.lam0

        self.initialize()

    def initialize(self):
        hammer_model = hammer.Harmonic(H=self.H,tau=self.tau)
        hammer_model.make_transition()
        Gamma,pi = hammer_model.transition,hammer_model.initial
        self.coeffs = hammer_model.k
        self.states = hammer_model.S

        tau_ev = 10
        sigma_ev = np.sqrt(tau_ev)
        self.a0 = 4
        self.b0 = self.a0 * sigma_ev**2.0

        self.hmm = HMM(states=self.states,parameter_sampling=False)
        self.hmm._parameters['Gamma'] = Gamma
        self.hmm._parameters['pi'] = pi
        self._parameters['x'] = np.ones(1)*50.0
        self._parameters['cov'] = np.array(7.0)

    def sample_cov(self,data: 'Data', z):
        h_idx = z < self.H
        y = data.output[h_idx].ravel()
        k = self.coeffs[z[h_idx]].ravel()
        mu = k*self.x
        eps = y - mu
        N = len(y)

        a = self.a0 + 1/2*N
        b = self.b0 + 1/2*eps.T @ eps

        self._parameters['cov'] = 1/gamma.rvs(a=a,scale=1/b)

    def sample_x(self,data:'Data',z):
        h_idx = z < self.H
        y = data.output[h_idx].ravel()
        k = self.coeffs[z[h_idx]].ravel()

        lam = (k.T @ k) / self.cov + self.lam0
        ell = (k.T @ y) / self.cov + self.ell0
        mu = ell / lam
        sigma = 1/lam
        self._parameters['x'] = mvn.rvs(mu,sigma).ravel()

    def loglikelihood(self,data):

        logl = np.zeros((len(data),self.states))
        xmu = data.output - self.coeffs[None,:] * self.x
        logl[:,:self.H] = -0.5*(np.log(2*np.pi*self.cov) + xmu**2.0 / self.cov)
        logl[:,self.H:] = -0.5*(np.log(2*np.pi*self.var_residual))
        return logl

    def forward(self,data:'Data'):
        logl = self.loglikelihood(data=data)
        self.hmm(logl)
        self.sample_x(data,z=self.hmm.z)
        self.sample_cov(data,z=self.hmm.z)


#%%
np.random.seed(123)
trials = 1
for trial in range(trials):
    f0 = np.random.uniform(35,1000,1)
    N = 20
    y = 1.0*(np.arange(N)+1)*f0
    y += np.random.normal(0,5,y.shape)
    noise = np.random.uniform(0,4000,N)
    y[N//2:] = noise[N//2:]
    y = np.sort(y)
    y = y[:,None]

    t = np.arange(y.shape[0])
    data = Data(y,time=t)

    model = Harmonic(tau=2)
    sampler = Gibbs()

    sampler.fit(data=data,model=model,samples=200)
    sampler.get_estimates(burn_rate=.98)

    z_hat = sampler._estimates['hmm.z']
    x_hat = sampler._estimates['x']

    print("targ = {}\nesti = {}".format(f0,x_hat))
    print(sampler._estimates['cov']**.5)

# %%
colors = get_colors()
labels = (z_hat>=model.H).astype(int)
plt.scatter(data.time*0+labels,data.output,c=colors[labels],s=10)
plt.ylim(0)
plt.grid()

chain = sampler.get_chain(burn_rate=0)

fig,ax = plt.subplots(2,figsize=(4,4))
for ii,p in enumerate(['x','cov']):
    _x = chain[p]
    _x = _x.reshape(_x.shape[0],-1)
    ax[ii].plot(_x,'k',alpha=.5)
    ax[ii].set_title(p)
plt.tight_layout()

# %%
