#%%
from gibbs import Gibbs, get_colors, get_scatter_kwds, plot_cov_ellipse, Module, Data, HMM, tqdm,scattercat, mvn, gamma, Plate, Mixture
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

    def sample_x(self,data,z):
        h_idx = z < self.H
        y = data.output[h_idx].ravel()
        k = self.coeffs[z[h_idx]].ravel()

        lam0 = (.2e-2)**2.0
        m0 = 500.0
        ell0 = m0*lam0

        lam = (k.T @ k) / self.cov + lam0
        ell = (k.T @ y) / self.cov + ell0
        mu = ell / lam
        sigma = 1/lam
        self._parameters['x'] = mvn.rvs(mu,sigma).ravel()

    def loglikelihood(self,data,mask=None):
        if mask is None:
            mask = np.ones(len(data)).astype(bool)
        logl = np.zeros((len(data),self.states))
        xmu = data.output - self.coeffs[None,:] * self.x
        logl[:,:self.H] = -0.5*(np.log(2*np.pi*self.cov) + xmu**2.0 / self.cov)
        logl[:,self.H:] = -0.5*(np.log(2*np.pi*self.var_residual))

        logl[mask] = 0
        return logl

    def logjoint(self,data):
        logl = self.loglikelihood(data) 
        logl = logl[np.arange(logl.shape[0]),self.hmm.z]
        return logl

    def forward(self,data:'Data',mask=None):
        logl = self.loglikelihood(data=data,mask=mask)
        self.hmm(logl)
        self.sample_x(data,z=self.hmm.z)
        self.sample_cov(data,z=self.hmm.z)


class HEMM(Module):
    r'''
        Finite Bayesian mixture of harmonic sources.

        Author: Julian Neri, 2022
    '''
    def __init__(self,tau=2,components=3):
        super(HEMM,self).__init__()
        self.components = components

        self.env = Plate(*[Harmonic(tau=tau) for i in range(self.components)])
        self.mix = Mixture(components=self.components)

    def loglikelihood(self,data: 'Data'):
        loglike = np.zeros((len(data),self.components))
        for k in range(self.components):
            loglike[:,k] = self.env[k].logjoint(data)
        return loglike

    def logjoint(self,data: 'Data'):
        logl = self.loglikelihood(data)
        temp = logl[np.arange(temp.shape[0]),self.mix.z]
        return temp

    def forward(self,data: 'Data'):
        if self.mix.z.shape[0] != len(data):
            self.mix._parameters['z'] = np.random.randint(0,self.components,len(data))
        for i in range(self.components):
            idx = self.mix.z == i
            self.env[i](data,mask=idx)
        self.mix(self.loglikelihood(data))
        

#%%
np.random.seed(123)
f0 = np.random.uniform(35,1000,2)
N = 20
y = 1.0*np.arange(1,N+1)[:,None]*f0[None,:]
y = y.ravel()
y += np.random.normal(0,2,y.shape)
y = np.sort(y)
y = y[:,None]

t = np.arange(y.shape[0])
data = Data(y,time=t)

# data.plot()

#%%
model = HEMM(components=2,tau=5)
sampler = Gibbs()

#%%
sampler.fit(data=data,model=model,samples=200)

#%%
sampler.get_estimates(burn_rate=.9)
s_hat = sampler._estimates['mix.z']
z_hat = sampler._estimates['env.0.hmm.z']
x_hat = np.concatenate([sampler._estimates['env.{}.x'.format(k)] for k in range(model.components)])
plt.plot(z_hat)
#%%
chain = sampler.get_chain(burn_rate=.25)
for k in range(model.components):
    plt.plot(chain['env.{}.x'.format(k)],alpha=.5);

# %%
colors = get_colors()
plt.scatter(data.time*0+s_hat,data.output,c=colors[s_hat],s=10)
plt.ylim(0)
plt.grid()
# %%

print("targ = {}\nesti = {}".format(f0,x_hat))

print(sampler._estimates['env.0.cov'])

# %%
