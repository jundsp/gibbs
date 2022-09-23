#%%
import numpy as np
from scipy.stats import multivariate_t, norm
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from .core import GibbsDirichletProcess, DirichletProcess
from .utils import plot_cov_ellipse

class DP_GMM(GibbsDirichletProcess):
    r'''
    Infinite Gaussian mixture model.

    Dirichlet process, gibbs sampling. 

    Author: Julian Neri, 2022

    Examples
    --------
    Import package
    >>> from gibbs import DP_GMM

    Create model and generate data from it
    >>> model = DP_GMM(output_dim=2)
    >>> x = model.generate(200)[0]

    Fit the model to the data using the Gibbs sampler.
    >>> model.fit(x,samples=20)
    >>> model.plot()
    >>> model.plot_samples()

    '''
    def __init__(self,output_dim=1,alpha=1,learn=True,outliers=False):
        super().__init__(alpha=alpha,learn=learn)
        self.outliers = outliers
        self.output_dim = output_dim
        self._hyperparameters = []
        self._kinds = []
        self._init_prior()
  
    def _init_prior(self):
        self._hyperparameter_lookup = [[],[]]
        

        m0 = np.zeros(self.output_dim)
        nu0 = (self.output_dim+1)
        k0 = .01
        S0 = np.eye(self.output_dim)*nu0
        self._hyperparameter_lookup[0] = tuple([m0,k0,S0,nu0])

        m0 = np.zeros(self.output_dim)
        nu0 = self.output_dim+1
        k0 = .01
        S0 = np.eye(self.output_dim)*nu0*.1
        self._hyperparameter_lookup[1] = tuple([m0,k0,S0,nu0])

    def _hyperparameter_sample(self):

        m0 = np.zeros(self.output_dim)
        nu0 = self.output_dim+1
        k0 = .01
        sigma = 1/np.random.gamma(1,5)
        S0 = np.eye(self.output_dim)*nu0*sigma
        
        return m0,k0,S0,nu0

        
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

    def _predictive_parameters(self,m,k,S,nu):
        _m = m
        _nu = nu - self.output_dim + 1.0
        _S = (k + 1.0)/(k*_nu) * S
        return _m, _S, _nu

    # Multivariate t
    def _predictive(self,x,m,k,S,nu):
        _m, _S, _nu = self._predictive_parameters(m,k,S,nu)
        return multivariate_t.pdf(x,loc=_m,shape=_S,df=_nu)

    def _sample_z_one(self,n):
        denominator = self.N + self._parameters['alpha'] - 1
        K_total = self.K + 1
        if self.outliers: K_total += 1

        rho = np.zeros(K_total)
        for k in range(self.K):
            idx = self._parameters['z'] == k
            idx[n] = False
            Nk = idx.sum() 

            if Nk > 0:
                m0,k0,S0,nu0 = self._hyperparameters[k]
                rho[k] = self._posterior_predictive(self.x[n],idx,m0,k0,S0,nu0) * Nk / denominator

        
        theta_prior = [[],[]]
        if self.outliers:
            m0,k0,S0,nu0 = self._hyperparameter_lookup[0]
            theta_prior[0] = tuple([m0,k0,S0,nu0])
            rho[-2] = self._prior_predictive(self.x[n],m0,k0,S0,nu0) * self._parameters['alpha'] / denominator

        m0,k0,S0,nu0 = self._hyperparameter_lookup[1]
        theta_prior[1] = tuple([m0,k0,S0,nu0])
        rho[-1] = self._prior_predictive(self.x[n],m0,k0,S0,nu0) * self._parameters['alpha'] / denominator

        rho /= rho.sum()
        _z_now = np.random.multinomial(1,rho).argmax()
        
        if _z_now > (self.K-1):
            kind = 1
            if self.outliers:
                if _z_now == self.K:
                    kind = 0
            _z_now = self.K + 0
            self._hyperparameters.append(theta_prior[kind])
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
        sigmas = np.random.gamma(shape=3,scale=.1,size=100)
        sigmas[0] = 2
        Sigma = np.stack([np.eye(self.output_dim)*s for s in sigmas],0)

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
                if self._kinds[k] == 0:
                    linestyle = 'dashed'
                else:
                    linestyle = 'solid'
                idx = z_hat==k
                m0,k0,S0,nu0 = self._hyperparameters[k]
                muN,kN,SN,nuN = self._posterior(idx,m0,k0,S0,nu0)
                mu_x, S_x, nu_x = self._predictive_parameters(muN,kN,SN,nuN)
                S_x *= nu_x/(nu_x - 2)

                plot_cov_ellipse(mu_x,S_x,facecolor='none',edgecolor=colors[k],linestyle=linestyle)
        elif self.output_dim == 1:
            _x = np.linspace(self.x.min(),self.x.max(),128)
            plt.scatter(self.x,self.x*0,c=colors[z_hat])
            for k in (K_hat):
                idx = z_hat==k
                m0,k0,S0,nu0 = self._hyperparameters[k]
                muN,kN,SN,nuN = self._posterior(idx,m0,k0,S0,nu0)
                mu_x, S_x, nu_x = self._predictive_parameters(muN,kN,SN,nuN)
                S_x *= nu_x/(nu_x - 2)
                _px = norm.pdf(_x,mu_x.ravel(),S_x.ravel())
                plt.plot(_x,_px,color=colors[k])
        plt.grid()
        plt.tight_layout()
