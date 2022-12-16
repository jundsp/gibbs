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
from .parameters import NormalWishart
from .plate import TimePlate, Plate
from ..dataclass import Data

#* Parameters should have a "sample /  learn" setting do register into the sampler. If not, then dont add to the chain, and allow for easy setting.

class StateSpace(Module):
    def __init__(self,output_dim=1,state_dim=2,hyper_sample=True,full_covariance=True):
        super(StateSpace,self).__init__()
        self._dimy = output_dim
        self._dimx = state_dim
        self.hyper_sample = hyper_sample
        self.full_cov = full_covariance
        self.initialize()
    
    def initialize(self):
        self.sys = NormalWishart(output_dim=self.state_dim, input_dim=self.state_dim,hyper_sample=self.hyper_sample,full_covariance=self.full_cov,sigma_ev=.01)
        self.obs = NormalWishart(output_dim=self.output_dim, input_dim=self.state_dim,hyper_sample=self.hyper_sample,full_covariance=self.full_cov,sigma_ev=1)
        self.pri = NormalWishart(output_dim=self.state_dim, input_dim=1,hyper_sample=self.hyper_sample,full_covariance=self.full_cov)

        self.I = np.eye(self.state_dim)

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
    def A(self):
        return self.sys.A
    @property
    def Q(self):
        return self.sys.Q
    @property
    def C(self):
        return self.obs.A
    @property
    def R(self):
        return self.obs.Q
    @property
    def m0(self):
        return self.pri.A.ravel()
    @property
    def P0(self):
        return self.pri.Q

    def forward(self,y,x,x_state):

        _y = y
        _x = x
        _x2 = x_state[1:]
        _x1 = x_state[:-1]
        _x0 = x_state[[0]]

        self.obs(y=_y,x=_x)
        self.sys(y=_x2,x=_x1)
        self.pri(y=_x0)


class LDS(Module):
    r'''
        Bayesian linear dynamical system.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,state_dim=2,states=1,parameter_sampling=True,hyper_sample=True,full_covariance=True):
        super(LDS,self).__init__()
        self._dimy = output_dim
        self._dimx = state_dim
        self._dimz = states
        self.parameter_sampling = parameter_sampling
        self.hyper_sample = hyper_sample
        self.full_cov = full_covariance

        self._parameters["x"] = np.zeros((1,state_dim))
        self.initialize()

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

    def A(self,k):
        return self.theta[k].A
    def Q(self,k):
        return self.theta[k].Q
    def C(self,k):
        return self.theta[k].C
    def R(self,k):
        return self.theta[k].R
    def m0(self,k):
        return self.theta[k].m0
    def P0(self,k):
        return self.theta[k].P0

    def initialize(self):
        self.theta = TimePlate(*[StateSpace(output_dim=self.output_dim,state_dim=self.state_dim,hyper_sample=self.hyper_sample,full_covariance=self.full_cov) for i in range(self.states)])
        self.I = np.eye(self.state_dim)
        
    def transition(self,x,z:int=0):
        return self.A(z) @ x
        
    def emission(self,x,z:int=0):
        return self.C(z) @ x
    
    def predict(self,mu,V,z:int=0):
        m = self.transition(mu,z=z)
        P = self.A(z) @ V @ self.A(z).T+ self.Q(z)
        return m, (P)

    def update(self,y,m,P,z:int=0):
        if (y.ndim == 1):
            return self.update_single(y,m,P,z=z)
        else:
            return self.update_multiple(y,m,P,z=z)

    def update_single(self,y,m,P,z:int=0):
        y_hat = self.emission(m,z=z)
        Sigma_hat = self.C(z) @ P @ self.C(z).T + self.R(z)
        K = la.solve(Sigma_hat, self.C(z) @ P).T
        mu = m + K @ (y - y_hat)
        V = (self.I - K @ self.C(z)) @ P
        return mu, (V)

    def update_multiple(self,y,m,P,z:int=0):
        CR = self.C(z).T @ la.inv(self.R(z))
        CRC = CR @ self.C(z)

        Lam = la.inv(P)
        ell = Lam @ m
        if len(y) > 0:
            ell += CR @ y.sum(0)
            Lam += CRC * y.shape[0]
        V = la.inv(Lam)
        mu = V @ ell
        return mu, (V)

    def _forward(self,data:'Data',z):
        self.T = data.T
        mu = np.zeros((self.T,self.state_dim))
        V = np.zeros((self.T,self.state_dim,self.state_dim))
        m = self.m0(z[0]).copy()
        P = self.P0(z[0]).copy()
        for n in range(self.T):
            ''' update'''
            y = data.output[data.time==n]
            mu[n], V[n] = self.update(y,m,P,z=z[n])
            ''' predict '''
            if n < self.T-1:
                m,P = self.predict(mu[n], V[n], z=z[n+1])
        return mu, V

    def _backward(self,mu,V,z):
        self._parameters['x'][-1] = mvn.rvs(mu[-1],V[-1])
        for t in range(self.T-2,-1,-1):
            state = z[t+1]
            m = self.A(state) @ mu[t]
            P = self.A(state) @ V[t] @ self.A(state).T + self.Q(state)
            K_star = V[t] @ self.A(state).T @ la.inv(P)

            _mu = mu[t] + K_star @ (self.x[t+1] - m)
            _V = (self.I - K_star @ self.A(state)) @ V[t]
            self._parameters['x'][t] = mvn.rvs(_mu,_V)

    def sample_x(self,data:'Data',z):
        mu, V = self._forward(data,z)
        self._backward(mu,V,z)

    def loglikelihood(self,data:'Data',z=None):
        z = self._check_input(data,z)
        logl = np.zeros((len(data)))
        for t in range(data.T):
            state = z[t]
            mu = self.C(state) @ self.x[t]
            Sigma = self.R(state)
            idx = data.time==t
            logl[idx] = mvn_logpdf(data.output[idx],mu,Sigma)
        return logl
        
    def _check_input(self,data:'Data',z=None):
        if z is None:
            z = np.zeros(data.T).astype(int)
        if z.shape[0] != data.T:
            raise ValueError("1st dim of z and y must be equal.")
        return z

    def forward(self,data:'Data',z=None):
        z = self._check_input(data,z)
        if self.x.shape[0] != data.T:
            self._parameters['x'] = np.zeros((data.T,self.state_dim))
        self.sample_x(data,z=z)
        if self.parameter_sampling == True:
            for i,m in enumerate(self.theta):
                m(y=data.output,x=self.x[data.time],x_state=self.x)
        