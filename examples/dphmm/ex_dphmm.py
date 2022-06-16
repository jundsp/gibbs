#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm,uniform,multivariate_t
from scipy.special import logsumexp
import matplotlib as mpl
import scipy.linalg as la
mpl.rcParams['figure.dpi'] = 150

from gibbs import GibbsDirichletProcess

'''
Infinite harmonic regression mixture model.

Dirichlet process, gibbs sampling. 

Groups sinusoid data based on their harmonic source.

# TODO: Handle skips in harmonics, and missing f0. Try on real data.

Author: Julian Neri, 2022
'''

class DP_Harmonic(GibbsDirichletProcess):
    def __init__(self):
        super().__init__()
        self.state_dim = 1
        self._init_prior()

    def _init_prior(self):
        m = np.zeros(self.state_dim)
        Lam = np.eye(self.state_dim)*1e-3
        rho_ev = 1/.1**2.0
        a = 5
        b = a / rho_ev
        self._theta_prior = dict(m=m,Lam=Lam,a=a,b=b)

    # Posterior of mv normal - gamma with mv normal likelihood.
    def _posterior(self,idx):
        N = idx.sum()
        y = np.sort(self.y[idx])
        Phi = np.arange(N)+1
        Phi = np.atleast_2d(Phi).reshape(-1,1)
        Lam = self._theta_prior['Lam']
        m = self._theta_prior['m']
        a = self._theta_prior['a']
        b = self._theta_prior['b']

        ell_hat = Lam @ m + y.conj().T @ Phi
        Lam_hat = Lam + Phi.conj().T @ Phi

        V = la.inv(Lam_hat)
        mu = V @ ell_hat

        a_hat = a + 0.5*N
        b_hat = b + 0.5*(y.conj().T @ y + m.conj().T @ Lam @ m - mu.conj().T @ Lam_hat @ mu).real

        theta = dict(m=mu,V=V,Lam=Lam_hat,a=a_hat,b=b_hat)
        for t in theta:
            theta[t] = theta[t].real
        return theta

    def _posterior_predictive(self,i,idx):
        theta_post = self._posterior(idx)
        h = len(idx)+1
        return self._predictive(i,theta_post,h=h)

    def _prior_predictive(self,i):
        return self._predictive(i,self._theta_prior,h=1.0)

    # Multivariate t
    def _predictive_parameters(self,theta,h):
        h = np.atleast_2d(np.asarray(h))
        m,Lam,a,b = theta['m'],theta['Lam'],theta['a'],theta['b']
        mu_y = h @ m
        V = la.inv(Lam)
        I = np.eye(h.shape[0])
        Sigma_y = b/a * (h @ V @ h.conj().T + I).real
        nu_y = 2*a.real
        return mu_y, Sigma_y, nu_y

    def _predictive(self,i,theta,h):
        mu_y,Sigma_y,nu_y = self._predictive_parameters(theta,h)
        return multivariate_t.logpdf(self.y[i],loc=mu_y,shape=Sigma_y,df=nu_y)
    
    def _sample_one_z(self,n):
        rho = np.zeros(self.K+1)
        for k in range(self.K):
            idx = (self._parameters['z'] == k)
            idx[n] = False
            count = idx.sum()
            if count > 0:
                rho[k] = self._posterior_predictive(n,idx) + np.log( count ) 

        rho[-1] = self._prior_predictive(n) + np.log( self._parameters['alpha'] )
        rho -= logsumexp(rho)
        rho = np.exp(rho)
        rho /= rho.sum()
        _z_now = np.random.multinomial(1,rho).argmax()
        if _z_now > (self.K-1):
            self.K += 1
        
        self._parameters['z'][n] = _z_now

    def _sample_z(self):
        for n in range(self.N):
            self._sample_one_z(n)

        self._collapse_groups()
        
    def fit(self, y, samples=100, burn_rate=0.5):
        self.y = y.copy()
        self.N = y.shape[0]
        if self._parameters['z'] is None:
            self._parameters['z'] = np.zeros(self.N).astype(int) - 1
            self.K = 0
        return super().fit(y, samples, burn_rate)

    def plot(self):
        z_hat = self._parameters['z']
        colors = np.array(['r','b','g','y','m','k']*20)
        plt.scatter(self.y,self.y*0+z_hat,c=colors[z_hat],alpha=.5)
        plt.xlim(0)


#%%
np.random.seed(123)
S = 1
K  = 10
k = np.arange(K)+1
nu = np.random.uniform(100,1000,S)
x = nu[:,None] * k[None,:]
x = x.ravel()
x += np.random.normal(0,5,x.shape)
x = np.sort(x)
plt.plot(x,x*0,'.')
plt.xlim(0), plt.grid()

# %%
model = DP_Harmonic()

# %%
model.fit(x,samples=100)

# %%
model.plot()
model.plot_samples()

# %%
