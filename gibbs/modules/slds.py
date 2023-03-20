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
from ..dataclass import Data

#* Parameters should have a "sample /  learn" setting do register into the sampler. If not, then dont add to the chain, and allow for easy setting.

class SLDS(Module):
    r'''
        Bayesian switching linear dynamical system.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,state_dim=2,states=1,learn_lds=True,learn_hmm=True,hyper_sample=True,full_covariance=True,expected_duration=10,circular=True):
        super(SLDS,self).__init__()
        self._dimy = output_dim
        self._dimx = state_dim
        self._dimz = states
        self.learn_lds = learn_lds
        self.learn_hmm = learn_hmm
        self.hyper_sample = hyper_sample
        self.full_cov = full_covariance
        self.expected_duration = expected_duration
        self.circular = circular 

        self.initialize()

    def initialize(self):
        self.hmm = HMM(states=self.states,expected_duration=self.expected_duration,parameter_sampling=self.learn_hmm,circular=self.circular)
        self.lds = LDS(output_dim=self.output_dim,state_dim=self.state_dim,states=self.states,full_covariance=self.full_cov,parameter_sampling=self.learn_lds,hyper_sample=self.hyper_sample)

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

    def logjoint(self,data:'Data'):
        logl = np.zeros((data.T,self.states))
        for s in range(self.states):
            mu = self.lds.x @ self.lds.C(s).T
            Sigma = self.lds.R(s)
            for t in np.unique(data.time):
                idx = t == data.time
                logl[t,s] = mvn.logpdf(data.output[idx],mu[t],Sigma).sum(0)

            m = self.lds.m0(s)
            P = self.lds.P0(s)
            logl[0,s] += mvn.logpdf(self.lds.x[0],m,P)
 
            m = self.lds.x[:-1] @ self.lds.A(s).T 
            P = self.lds.Q(s)

            logl[1:,s] += mvn_logpdf(self.lds.x[1:],m,P)
        return logl

    def loglikelihood(self,y,mask):
        return self.lds.loglikelihood(y,z=self.hmm.z,mask=mask)

    def forward(self,data:'Data'):
        if self.hmm.z.shape[0] != data.T:
            self.hmm._parameters['z'] = np.random.randint(0,self.states,data.T)
            self.hmm(logl=np.zeros((data.T,self.states)))
        else:
            self.hmm(logl=self.logjoint(data))

        self.lds(data,z=self.hmm.z)
        


