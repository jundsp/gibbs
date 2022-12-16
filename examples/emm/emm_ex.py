#%%
from gibbs import Gibbs, Module, Data, la, mvn, gamma, Plate, Mixture,mvn_logpdf,get_colors,get_scatter_kwds
import numpy as np
import os
import matplotlib.pyplot as plt
import sines
import soundfile as sf

class Envelope(Module):
    r'''
        Bayesian spectral envelope model (regression)

        Author: Julian Neri, 2022
    '''
    def __init__(self,order=5):
        super(Envelope,self).__init__()

        self.order = order
        alpha0 = np.ones(order)
        alpha0[0] = 1e-2
        alpha0[1] = 1e-2
        sigma_ev = 1
        self.a0 = 1
        self.b0 = self.a0 * sigma_ev**2.0

        self.c0 = np.ones(order)
        self.d0 = self.c0 / alpha0

        self._parameters['cov'] = np.atleast_1d(.1)
        self._parameters['x'] = np.random.normal(0,.5,order)
        self._parameters['alpha'] = alpha0 + 0

        self.env = sines.Envelope(order=order,looseness=.66)

    def sample_x(self,data: 'Data'):
        Phi = self.env(data.input[:,0])
        Lam = (Phi.T @ Phi) / self.cov + np.diag(self.alpha)
        ell = (Phi.T @ data.output[:,0]) / self.cov
        Sigma = la.inv(Lam)
        mu = Sigma @ ell
        self._parameters['x'] = mvn.rvs(mu,Sigma)

    def sample_cov(self,data: 'Data'):
        Phi = self.env(data.input[:,0])
        eps = data.output[:,0] - Phi @ self.x

        a = self.a0 + 1/2*len(data)
        b = self.b0 + 1/2*eps.T @ eps

        self._parameters['cov'] = 1/gamma.rvs(a=a,scale=1/b)
    
    def sample_alpha(self):
        c = self.c0 + 1/2
        d = self.d0 + 1/2*(self.x ** 2.0)
        self._parameters['alpha'] = gamma.rvs(a=c,scale=1/d)

    def loglikelihood(self,data:'Data'):
        Phi = self.env(data.input[:,0])
        y_eps = data.output[:,0] - Phi @ self.x
        quad = y_eps ** 2.0 / self.cov
        return -0.5*(np.log(2*np.pi*self.cov) + quad)

    def forward(self,data: 'Data'):
        self.sample_x(data)
        self.sample_cov(data)
        self.sample_alpha()


class EMM(Module):
    r'''
        Finite Bayesian mixture of Envelopes.

        Author: Julian Neri, 2022
    '''
    def __init__(self,order=1,components=3):
        super(EMM,self).__init__()
        self.order = order
        self.components = components

        self.env = Plate(*[Envelope(order=self.order) for i in range(self.components)])
        self.mix = Mixture(components=self.components)

    def loglikelihood(self,data: 'Data'):
        loglike = np.zeros((len(data),self.components))
        for k in range(self.components):
            loglike[:,k] = self.env[k].loglikelihood(data)
        return loglike

    def logjoint(self,data: 'Data'):
        logl = self.loglikelihood(data)
        temp = logl[np.arange(temp.shape[0]),self.mix.z]
        return temp

    def forward(self,data: 'Data'):
        self.mix(self.loglikelihood(data))
        for i in range(self.components):
            idx = self.mix.z == i
            data_temp = Data(y=data.output[idx],x=data.input[idx])
            self.env[i](data_temp)


# np.random.seed(1)
filename = "source_example"
audio,sr = sf.read("/Users/julian/Documents/MATLAB/sounds/{}.wav".format(filename))
audio += np.random.normal(0,1e-2,audio.shape)

time = len(audio)//2 / sr
time = time + np.arange(5)*.01
features = sines.short_term(audio,sr,window_size=32,time=time,confidence=.9,resolutions=1)

y = np.log(np.array(features['amplitude']))[:,None] 
x = np.array(features['frequency'])[:,None]/sr*2
data = Data(y=y,x=x)

#%%
order = 11
model = EMM(order=order,components=5 )
sampler = Gibbs()

#%%
sampler.fit(data,model,500)

#%%
sampler.get_estimates('median',burn_rate=.9)
chain = sampler.get_chain(burn_rate=.9)

z_hat = sampler._estimates['mix.z'].astype(int)

x_hat = np.stack([sampler._estimates['env.{}.x'.format(i)] for i in range(model.components)],-1)
cov_hat = np.stack([sampler._estimates['env.{}.cov'.format(i)] for i in range(model.components)],-1)
alpha_hat = np.stack([sampler._estimates['env.{}.alpha'.format(i)] for i in range(model.components)],-1)

xx = np.linspace(0,1,512)
Phi = model.env[0].env(xx)
y_hat = Phi @ x_hat

#%%
plt.figure(figsize=(4,3))
colors = get_colors()
plt.scatter(x.ravel(),y.ravel(),c=colors[z_hat],**get_scatter_kwds())
for k in np.unique(z_hat):
    plt.plot(xx,y_hat[:,k],color=colors[k])
plt.xlim(0,1)
plt.ylim(y.min()-1,y.max()+1)
plt.xlabel('frequency (normalized)')
plt.ylabel('log magnitude')
plt.tight_layout()
# plt.show()
# plt.savefig('emm_source_example_M16.pdf')
print(cov_hat)
print(alpha_hat)

fig,ax = plt.subplots(len(chain),figsize=(3,len(chain)))
for ii,p in enumerate(chain):
    _x = chain[p]
    _x = _x.reshape(_x.shape[0],-1)
    ax[ii].plot(_x,'k',alpha=.1)
    ax[ii].set_title(p)
plt.tight_layout()

# %%
