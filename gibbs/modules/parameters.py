from typing import OrderedDict,overload,Optional, Iterable, Set
from itertools import islice
import operator
import numpy as np

from scipy.stats import multivariate_normal as mvn
from scipy.stats import gamma, wishart, dirichlet
from scipy.special import logsumexp
import scipy.linalg as la

from ..utils import mvn_logpdf, makesymmetric
from .module import Module

class NormalWishart(Module):
    r'''
        Normal wishart system parameters

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,input_dim=1,hyper_sample=True,full_covariance=True,sigma_ev=1):
        super(NormalWishart,self).__init__()
        self._dimo = output_dim
        self._dimi = input_dim
        self.hyper_sample = hyper_sample
        self.full_covariance = full_covariance
        self.sigma_ev = sigma_ev

        self.initialize()

    @property
    def output_dim(self):
        return self._dimo
    @output_dim.setter
    def output_dim(self,value):
        value = np.maximum(1,value)
        if value != self._dimo:
            self._dimo = value
            self.initialize()

    @property
    def input_dim(self):
        return self._dimi
    @input_dim.setter
    def input_dim(self,value):
        value = np.maximum(1,value)
        if value != self.dimi:
            self._dimi = value
            self.initialize()

    def initialize(self):
        self.define_priors()
        self.initialize_parameters()
        
    def define_priors(self):
        self.nu0 = self.output_dim + .1
        s = self.nu0*self.sigma_ev**2.0
        self.iW0 = np.eye(self.output_dim)*s

        self.a0 = self.nu0 / 2.0
        self.b0 = 1.0 / (2*s)

        self.c0 = .5
        self.d0 = .5
        
    def initialize_parameters(self):
        A = np.eye(self.output_dim,self.input_dim)
        A += np.random.normal(0,1e-3,A.shape)
        A /= np.abs(A).sum(-1)[:,None]
        Q = np.eye(self.output_dim)
        alpha = np.ones((self.input_dim))*1e-1

        self._parameters["A"] = A.copy()
        self._parameters["Q"] = Q.copy()
        self._parameters["alpha"] = alpha.copy()

    def sample_A(self):
        tau = 1.0 / np.diag(self.Q)
        Alpha = np.diag(self.alpha)
        for row in range(self.output_dim):
            ell = tau[row] * (self.y[:,row] @ self.x)
            Lam = tau[row] *  ( (self.x.T @ self.x) + Alpha )
            Sigma = la.inv(Lam)
            mu = Sigma @ ell
            self._parameters['A'][row] = mvn.rvs(mu,Sigma)

    def sample_alpha(self):
        if self.hyper_sample is False:
            return 0
        # The marginal prior over A is then a Cauchy distribution: N(a,|0,1/alpha)Gam(alpha|.5,.5) => Cauchy(a)
        a = np.zeros(self.input_dim)
        b = np.zeros(self.input_dim)
        for col in range(self.input_dim):
            a[col] = self.c0 + 1/2*self.output_dim
            b[col] = self.d0 + 1/2*( ((self.A[:,col]) ** 2.0)).sum(0)
        self._parameters['alpha'] = np.atleast_1d(gamma.rvs(a=a,scale=1.0/b))

    def sample_Q(self):
        if self.full_covariance:
            Q = self._sample_cov_full()
        else:
            Q = self._sample_cov_diag()
        self._parameters['Q'] = Q

    def _sample_cov_diag(self):
        Nk = self.N + self.input_dim
        x_eps = self.y - self.x @ self.A.T
        quad = np.diag(x_eps.T @ x_eps) + np.diag((self.A) @ np.diag(self.alpha) @ (self.A).T)

        a_hat = self.a0 + 1/2 * Nk
        b_hat = self.b0 + 1/2 * quad

        tau = np.atleast_1d(gamma.rvs(a=a_hat,scale=1/b_hat))
        return np.diag(1/tau)

    def _sample_cov_full(self):
        nu = self.nu0 + self.N

        y_hat = (self.x @ self.A.T)
        x_eps = (self.y - y_hat)
        quad = x_eps.T @ x_eps
        iW = self.iW0 + quad
        iW = .5*(iW + iW.T)
        W = la.inv(iW)

        Lambda = np.atleast_2d(wishart.rvs(df=nu,scale=W) )
        return la.inv(Lambda)

    def loglikelihood(self,y,x=None,mask=None):
        x,mask = self._check_input(y,x,mask)
        mu = x @ self.A.T
        loglike = mvn_logpdf(y[mask],mu[mask],self.Q)
        return loglike

    def _check_input(self,y,x=None,mask=None):
        if y.ndim != 2:
            raise ValueError("y must be 2d")

        if x is None:
            x = np.ones((y.shape[0],self.input_dim))
        if x.ndim != 2:
            raise ValueError("x must be 2d")
        if y.shape[0] != x.shape[0]:
            raise ValueError("dim 1 of y and x must match")
        
        if mask is None:
            mask = np.ones(y.shape[0]).astype(bool)
        if mask.ndim != 1:
            raise ValueError("mask must be 1d")
        if y.shape[0] != mask.shape[0]:
            raise ValueError("mask must match y in dim 1")

        self.output_dim = y.shape[-1]
        self.state_dim = x.shape[-1]
        return x, mask


    def forward(self,y,x=None,mask=None):
        x,mask = self._check_input(y,x,mask)
        
        self.y = y[mask]
        self.x = x[mask]
        self.N = self.y.shape[0]

        self.sample_A()
        self.sample_Q()
        self.sample_alpha()



class CovarianceMatrix(Module):
    def __init__(self,dim=1) -> None:
        super(CovarianceMatrix,self).__init__()
        self.dim = dim
        self.nu0 = dim+1.0
        self.W0 = (np.eye(dim)) / self.nu0
        self.iW0 = la.inv(self.W0)
        Lambda = np.atleast_2d(wishart.rvs(df=self.nu0,scale=self.W0))
        self._parameters['cov'] = la.inv(Lambda)

    def sample_cov(self):
        nu = self.nu0 + self.N

        x_eps = (self.y - self.x)
        quad = x_eps.T @ x_eps
        iW = self.iW0 + quad
        W = la.inv( makesymmetric(iW))

        Lambda = np.atleast_2d(wishart.rvs(df=nu,scale=W))
        self._parameters['cov'] =  la.inv(Lambda)

    def _check_input(self,y,x=None,mask=None):
        if y.ndim != 2:
            raise ValueError("y must be 2d")

        if x is None:
            x = np.ones((y.shape[0],self.dim))
        if x.ndim != 2:
            raise ValueError("x must be 2d")
        if y.shape != x.shape:
            raise ValueError("dim of y and x must match")
        
        if mask is None:
            mask = np.ones(y.shape[0]).astype(bool)
        if mask.ndim != 1:
            raise ValueError("mask must be 1d")
        if y.shape[0] != mask.shape[0]:
            raise ValueError("mask must match y in dim 1")

        self.dim = y.shape[-1]
        return x, mask

    def forward(self,y,x=None,mask=None):
        x,mask = self._check_input(y,x,mask)
        
        self.y = y[mask]
        self.x = x[mask]
        self.N = self.y.shape[0]

        self.sample_cov()