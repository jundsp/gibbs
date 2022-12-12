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
from .lds import LDS
from .hmm import HMM

#* Parameters should have a "sample /  learn" setting do register into the sampler. If not, then dont add to the chain, and allow for easy setting.

class SLDS(Module):
    r'''
        Bayesian switching linear dynamical system.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,state_dim=2,states=1,parameter_sampling=True,hyper_sample=True,full_covariance=True,expected_duration=10):
        super(SLDS,self).__init__()
        self._dimy = output_dim
        self._dimx = state_dim
        self._dimz = states
        self.parameter_sampling = parameter_sampling
        self.hyper_sample = hyper_sample
        self.full_cov = full_covariance
        self.expected_duration = expected_duration

        self.initialize()

    def initialize(self):
        self.hmm = HMM(states=self.states,expected_duration=self.expected_duration,parameter_sampling=self.parameter_sampling)
        self.lds = LDS(output_dim=self.output_dim,state_dim=self.state_dim,states=self.states,full_covariance=self.full_cov,parameter_sampling=self.parameter_sampling,hyper_sample=self.hyper_sample)

    @property
    def output_dim(self):
        return self._dimy
    
    @output_dim.setter
    def output_dim(self,value):
        value = np.maximum(1,value)
        if value != self._dimy:
            self._dimy = value
            self.initialize()

    @property
    def state_dim(self):
        return self._dimx
    @state_dim.setter
    def state_dim(self,value):
        value = np.maximum(1,value)
        if value != self._dimx:
            self._dimx = value
            self.initialize()

    @property
    def states(self):
        return self._dimz
    @states.setter
    def states(self,value):
        value = np.maximum(1,value)
        if value != self._dimz:
            self._dimz = value
            self.initialize()

    def logjoint(self,y,mask):
        logl = np.zeros((self.T, self.states))
        logly = np.zeros((self.T,self.N))
        rz, cz = np.nonzero(mask)
        
        for s in range(self.states):
            mu =  self.lds.x @ self.lds.C(s).T
            Sigma = self.lds.R(s)
            logly[rz,cz] = mvn_logpdf(y[rz,cz],mu[rz],Sigma)
            logl[:,s] = logly.sum(1)

            m = self.lds.m0(s)[None,:]
            P = self.lds.P0(s)
            logl[0,s] += mvn_logpdf(self.lds.x[[0]],m,P)
 
            m = self.lds.x[:-1] @ self.lds.A(s).T 
            P = self.lds.Q(s)
            logl[1:,s] += mvn_logpdf(self.lds.x[1:],m,P)

        return logl

    def loglikelihood(self,y,mask):
        return self.lds.loglikelihood(y,z=self.hmm.z,mask=mask)

    def _check_data(self,y,mask=None):
        if y.ndim != 3:
            raise ValueError("input must be 3d")
        if mask is None:
            mask = np.ones(y.shape[:2]).astype(bool)
        if mask.ndim != 2:
            raise ValueError("mask must be 2d")
        if y.shape[:2] != mask.shape[:2]:
            raise ValueError("mask must match y in dimensions")
        self.T, self.N, self.output_dim = y.shape

        if self.hmm.z.shape[0] != self.T:
            self.hmm._parameters['z'] = np.random.randint(0,self.states,self.T)

        return mask

    def forward(self,y,mask=None):
        mask = self._check_data(y=y,mask=mask)
        self.lds(y=y,z=self.hmm.z,mask=mask)
        self.hmm(logl=self.logjoint(y,mask))


