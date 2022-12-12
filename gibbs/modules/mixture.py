from typing import OrderedDict,overload,Optional, Iterable, Set
from itertools import islice
import operator
import numpy as np

from scipy.stats import multivariate_normal as mvn
from scipy.stats import gamma, wishart, dirichlet
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
    def __init__(self,components=3):
        super().__init__()
        self.components = components

        self._parameters["z"] = np.ones((1,components)).astype(int)
        self._parameters["pi"] = np.ones(components)/components
        self.alpha0 = np.ones(components) 

    def sample_z(self,logl):
        rho = logl + np.log(self.pi)
        rho -= logsumexp(rho,-1).reshape(-1,1)
        rho = np.exp(rho)
        rho /= rho.sum(-1).reshape(-1,1)
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

    def forward(self,logl):
        self._check_data(logl)
        self.sample_z(logl)
        self.sample_pi()



class InfiniteMixture(Module):
    r'''
        Infinite Bayesian mixture.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,hyper_sampling=False):
        super().__init__()
        self.hyper_sampling = hyper_sampling

        self._parameters["z"] = np.ones((1,1)).astype(int)
        self._parameters["pi"] = np.ones(1)/1
        self.alpha0 = np.ones(1) 

    def sample_z(self,logl):
        rho = logl + np.log(self.pi)
        rho -= logsumexp(rho,-1).reshape(-1,1)
        rho = np.exp(rho)
        rho /= rho.sum(-1).reshape(-1,1)
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

    def forward(self,logl):
        self._check_data(logl)
        self.sample_z(logl)
        self.sample_pi()


