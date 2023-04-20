import numpy as np

from scipy.stats import multivariate_normal as mvn
from scipy.stats import dirichlet
from scipy.special import logsumexp
from .module import Module 
from .parameters import NormalWishart
from .plate import Plate
from ..dataclass import Data
np.seterr(divide='ignore')


#* Parameters should have a "sample /  learn" setting do register into the sampler. If not, then dont add to the chain, and allow for easy setting.

class HMM(Module):
    r'''
        Bayesian hidden Markov model.

        Gibbs sampling. 

        Author: Julian Neri, December 2022
    '''
    def __init__(self,states=1,expected_duration=1,parameter_sampling=True,circular=True):
        super().__init__()
        self._dimz = states
        self.expected_duration = expected_duration
        self.parameter_sampling = parameter_sampling
        self.circular = circular

        self._parameters["z"] = np.random.randint(0,self.states,1)
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

    def initialize(self):
        A_kk = self.expected_duration / (self.expected_duration+1)
        A_jk = 1.0
        if self.states > 1:
            A_jk = (1-A_kk) / (self.states-1)
        Gamma = np.ones((self.states,self.states)) * A_jk
        np.fill_diagonal(Gamma,A_kk)
        if self.circular is False:
            idx = 1-np.triu(np.ones(self.states))
            Gamma[idx == 1] = 0

        Gamma /= Gamma.sum(-1).reshape(-1,1)
        
        pi = np.ones(self.states) 
        pi /= pi.sum()

        self.Gamma0 = Gamma.copy()
        self.pi0 = pi.copy()

        self.set_parameters(Gamma,pi)
    
    def set_parameters(self,Gamma,pi):
        self._parameters["Gamma"] = Gamma.copy()
        self._parameters["pi"] = pi.copy()
        self.log_Gamma = np.log(Gamma)
        self.log_pi = np.log(pi)

    def _predict_hmm(self,alpha,transpose=False):
        if transpose:
            return np.log(np.exp(alpha) @ self.Gamma.T)
        else:
            return np.log(np.exp(alpha) @ self.Gamma)

    def _forward_hmm(self,logl):
        alpha = np.zeros((self.T,self.states))
        c = np.zeros((self.T))    
        prediction = np.log(self._parameters['pi']).reshape(1,-1)
        for t in range(self.T):
            alpha[t] = logl[t] + prediction
            c[t] = logsumexp(alpha[t])
            alpha[t] -= c[t]
            prediction = self._predict_hmm(alpha[t])
        return alpha

    def _backward_hmm(self,log_alpha:np.ndarray):
        beta = log_alpha[-1] - logsumexp(log_alpha[-1])
        self._parameters['z'][-1] = np.random.multinomial(1,np.exp(beta)).argmax()
        for t in range(self.T-2,-1,-1):
            beta = self.log_Gamma[:,self.z[t+1]] + log_alpha[t]
            beta -= logsumexp(beta)
            beta = np.exp(beta)
            self._parameters['z'][t] = np.random.multinomial(1,beta).argmax()
        
    def sample_z(self,logl):
        log_alpha = self._forward_hmm(logl)
        self._backward_hmm(log_alpha)

    def sample_Gamma(self):
        alpha = np.zeros(self.states)
        for k in range(self.states):
            n1 = (self.z[:-1] == k)
            for j in range(self.states):
                n2 = (self.z[1:] == j)
                alpha[j] = self.Gamma0[k,j] + np.sum(n1 & n2)
            nz = np.nonzero(alpha)
            self._parameters['Gamma'][k] = np.zeros(self.states)
            self._parameters['Gamma'][k,nz] = dirichlet.rvs(alpha[nz])

        self.log_Gamma = np.log(self.Gamma)

    def sample_pi(self):
        alpha = np.zeros(self.states)
        for k in range(self.states):
            alpha[k] = self.pi0[k] + (self.z[0] == k).sum()
        nz = np.nonzero(alpha)
        self._parameters['pi'] = np.zeros(self.states)
        self._parameters['pi'][nz] = dirichlet.rvs(alpha[nz]).ravel()

        self.log_pi = np.log(self.pi)

    def _check_data(self,logl):
        if logl.ndim != 2:
            raise ValueError("input must have 2d")
        self.T, states = logl.shape
        if states != self.states:
            raise ValueError("input logl must have same dimensonality as z (T x S)")
        if self.z.shape[0] != self.T:
            self._parameters['z'] = np.random.randint(0,self.states,self.T)

    def forward(self,logl):
        self._check_data(logl)
        self.sample_z(logl)
        if self.parameter_sampling:
            self.sample_Gamma()
            self.sample_pi()

class GHMM(Module):
    r'''
        Finite Bayesian mixture of Gaussians.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,states=3,hyper_sample=True,full_covariance=True):
        super(GHMM,self).__init__()
        self._dimy = output_dim
        self._dimz = states
        self.hyper_sample = hyper_sample
        self.full_covariance = full_covariance

        self.initialize()

    def initialize(self):
        self.theta = Plate(*[NormalWishart(output_dim=self.output_dim,hyper_sample=self.hyper_sample,full_covariance=self.full_covariance) for i in range(self.states)])
        self.hmm = HMM(states=self.states)

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
    def states(self):
        return self._dimz
    @states.setter
    def states(self,value):
        value = np.maximum(1,value)
        if value != self._dimz:
            self._dimz = value
            self.initialize()

    def loglikelihood(self,data:'Data'):
        logl = np.zeros((data.T,self.states))
        for k in range(self.states):
            logl[data.time,k] = mvn.logpdf(data.output,self.theta[k].A.ravel(),self.theta[k].Q)
        return logl

    def forward(self,data:'Data'):
        self.hmm(self.loglikelihood(data))
        for k in range(self.states):
            self.theta[k].forward(data.output[self.hmm.z[data.time]==k])
