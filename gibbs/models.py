#%%
import numpy as np
from scipy.stats import multivariate_t
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from .core import GibbsDirichletProcess, DirichletProcess
from .utils import plot_cov_ellipse

'''
Infinite Gaussian mixture model.

Dirichlet process, gibbs sampling. 

© Julian Neri, 2022
'''

class DP_GMM(GibbsDirichletProcess):
    def __init__(self,output_dim=1,alpha=1,learn=True):
        super().__init__(alpha=alpha,learn=learn)
        self.output_dim = output_dim
        self._hyperparameters = []
        self._kinds = []
        self._init_prior()
  
    def _init_prior(self):
        self._hyperparameter_lookup = []
        
        m0 = np.zeros(self.output_dim)
        nu0 = self.output_dim+1
        k0 = .01
        S0 = np.eye(self.output_dim)*nu0*.1
        self._hyperparameter_lookup.append(tuple([m0,k0,S0,nu0]))

        m0 = np.zeros(self.output_dim)
        nu0 = self.output_dim*2
        k0 = .01
        S0 = np.eye(self.output_dim)*nu0*50
        self._hyperparameter_lookup.append(tuple([m0,k0,S0,nu0]))

    def _sample_hyperparameters(self):
        j = np.random.randint(0,2)
        return j
        
    def _posterior(self,idx,m0,k0,S0,nu0):
        N = idx.sum()
        xk = np.atleast_2d(self.x[idx])
        x_bar = xk.mean(0)
        S = xk.T @ xk
        kN = k0 + N
        mN = (k0 * m0 + x_bar * N)/kN
        nuN = nu0 + N
        SN = S0 + S + k0*np.outer(m0,m0) - kN*np.outer(mN,mN)
        return mN,kN,SN,nuN

    def _posterior_predictive(self,x_i,idx,m0,k0,S0,nu0):
        m,k,S,nu = self._posterior(idx,m0,k0,S0,nu0)
        return self._predictive(x_i,m,k,S,nu)

    def _prior_predictive(self,x_i,m0,k0,S0,nu0):
        return self._predictive(x_i,m0,k0,S0,nu0)

    # Multivariate t
    def _predictive(self,x,m,k,S,nu):
        _m = m
        _nu = nu - self.output_dim + 1.0
        _S = (k + 1.0)/(k*_nu) * S
        return multivariate_t.pdf(x,loc=_m,shape=_S,df=_nu)

    def _sample_z_one(self,n):
        rho = np.zeros(self.K+1)
        for k in range(self.K):
            idx = self._parameters['z'] == k
            idx[n] = False
            if idx.sum() > 0:
                m0,k0,S0,nu0 = self._hyperparameters[k]
                rho[k] = self._posterior_predictive(self.x[n],idx,m0,k0,S0,nu0) * idx.sum()
        kind = self._sample_hyperparameters()
        m0,k0,S0,nu0 = self._hyperparameter_lookup[kind]
        rho[-1] = self._prior_predictive(self.x[n],m0,k0,S0,nu0) * self._parameters['alpha']
        rho /= rho.sum()
        _z_now = np.random.multinomial(1,rho).argmax()
        
        if _z_now > (self.K-1):
            self._hyperparameters.append(tuple([m0,k0,S0,nu0]))
            self._kinds.append(kind)
            self.K += 1
        self._parameters['z'][n] = _z_now
        
    def _sample_z(self):
        tau = np.random.permutation(self.N)
        for n in tau:
            self._sample_z_one(n)
        self._collapse_groups()

    def generate(self,n=100):
        dp = DirichletProcess(alpha=self._parameters['alpha'])
        mu = np.random.normal(0,2,(100,self.output_dim))
        Sigma = np.stack([np.eye(self.output_dim)*.1]*100,0)

        z = np.zeros(n).astype(int)
        x = np.zeros((n,self.output_dim))
        for i in range(n):
            z[i] = dp.sample()
            x[i] = np.random.multivariate_normal(mu[z[i]],Sigma[z[i]])
        return x, z

    def fit(self,x,samples=100):
        self.x = x
        self.N = x.shape[0]
        if x.shape[-1] != self.output_dim:
            self.output_dim = x.shape[-1]
            self._init_prior()
        if self._parameters['z'] is None:
            self._parameters['z'] = np.zeros(self.N).astype(int) - 1
            self.K = 0
        super().fit(samples)

    def plot(self):
        z_hat = self._parameters['z'].astype(int)
        K_hat = np.unique(z_hat)
        colors = np.array(['r','g','b','m','y','k','orange']*30)
        plt.figure(figsize=(5,4))
        if self.output_dim == 2:
            plt.scatter(self.x[:,0],self.x[:,1],c=colors[z_hat])
            for k in (K_hat):
                idx = z_hat==k
                m0,k0,S0,nu0 = self._hyperparameters[k]
                muN,kN,SN,nuN = self._posterior(idx,m0,k0,S0,nu0)

                plot_cov_ellipse(muN,SN/nuN,facecolor='none',edgecolor=colors[k])
        elif self.output_dim == 1:
            plt.scatter(self.x,self.x*0,c=colors[z_hat])
        plt.grid()
        plt.tight_layout()
