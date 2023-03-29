from typing import OrderedDict,overload,Optional, Iterable, Set
from itertools import islice
import operator
import numpy as np

from scipy.stats import multivariate_normal as mvn
from scipy.stats import gamma, wishart, dirichlet, multinomial, beta
from scipy.special import logsumexp
import scipy.linalg as la

from ..utils import mvn_logpdf
from .module import Module

class Mixture(Module):
    r'''
        Finite Bayesian mixture.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,components=3,learn=True):
        super().__init__()
        self.components = components
        self.learn = learn

        self._parameters["z"] = np.ones(1).astype(int)
        self._parameters["pi"] = np.ones(components)/components
        self._parameters['rho'] = np.ones((1,components)) / components
        self.alpha0 = np.ones(components) 

    def sample_z(self,logl):
        rho = logl + np.log(self.pi)
        rho -= logsumexp(rho,-1).reshape(-1,1)
        rho = np.exp(rho)
        rho /= rho.sum(-1).reshape(-1,1)
        self._parameters['rho'] = rho + 0
        for n in range(self.N):
            self._parameters['z'][n] = np.random.multinomial(1,rho[n]).argmax()

    def sample_pi(self):
        alpha = np.zeros(self.components)
        for k in range(self.components):
            alpha[k] = (self.z==k).sum() + self.alpha0[k]
        self._parameters['pi'] = dirichlet.rvs(alpha)

    def _check_data(self,logl):
        self.N, components = logl.shape
        if components != self.components:
            raise ValueError("input must have same dimensonality as z (N x C)")
        if self.z.shape[0] != logl.shape[0]:
            self._parameters['z'] = np.random.randint(0,self.components,(self.N))
            self._parameters['rho'] = np.ones((self.N,self.components)) / self.components

    def forward(self,logl):
        self._check_data(logl)
        self.sample_z(logl)
        if self.learn:
            self.sample_pi()



class InfiniteMixture(Module):
    r'''
        Infinite Bayesian mixture.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,alpha=1,learn=True):
        super().__init__()
        self.learn = learn

        # Control parameters of the prior
        self.a = 1
        self.b = 1

        self._parameters["eta"] = np.array([.5])
        self._parameters["alpha"] = np.atleast_1d(alpha)

    def sample_alpha(self,z):
        K = len(np.unique(z))
        
        b_hat = self.b - np.log(self.eta)
        y = self.a + K - 1
        z = self.N * b_hat
        pi_eta = y/(y+z)
        pi_ = np.array([pi_eta,1-pi_eta]).ravel()
        m = multinomial.rvs(1.0,pi_).argmax()
        if m == 0:
            a_hat = self.a + K
        else:
            a_hat = self.a + K - 1
        self._parameters['alpha'] = gamma.rvs(a=a_hat,scale=1/b_hat)

    def sample_eta(self):
        a = self.alpha + 1.0
        b = self.N
        self._parameters['eta'] = beta.rvs(a,b)

    def forward(self,z):
        z = z.ravel().astype(int)
        self.N = z.shape[0]
        if self.learn:
            self.sample_alpha(z)
            self.sample_eta()