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
        self.sys = NormalWishart(output_dim=self.state_dim, input_dim=self.state_dim,hyper_sample=self.hyper_sample,full_covariance=self.full_cov)
        self.obs = NormalWishart(output_dim=self.output_dim, input_dim=self.state_dim,hyper_sample=self.hyper_sample,full_covariance=self.full_cov)
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

    def _check_input(self,y,x,mask_time=None,mask_data=None):
        if y.ndim != 3:
            raise ValueError("y must be 3d")
        if x.ndim != 2:
            raise ValueError("x must be 2d")
        if y.shape[0] != x.shape[0]:
            raise ValueError("dim 1 of y and x must match")

        if mask_time is None:
            mask = np.ones(y.shape[0]).astype(bool)
        if mask_time.ndim != 1:
            raise ValueError("mask_time must be 1d")
        if y.shape[0] != mask_time.shape[0]:
            raise ValueError("mask_time must match y in dimensions")


        if mask_data is None:
            mask = np.ones(y.shape[:2]).astype(bool)
        if mask_data.ndim != 2:
            raise ValueError("mask_data must be 2d")
        if y.shape[:2] != mask_data.shape[:2]:
            raise ValueError("mask_data must match y in 2 dimensions")

        self.output_dim = y.shape[-1]
        self.state_dim = x.shape[-1]
        return mask_time, mask_data

    def forward(self,y,x,mask_time=None,mask_data=None):
        mask_time, mask_data = self._check_input(y,x,mask_time=mask_time,mask_data=mask_data)
        
        rz, cz = np.nonzero(mask_data & mask_time[:,None])

        _y = y[rz,cz]
        _x = x[rz]
        _x2 = x[1:][mask_time[1:]]
        _x1 = x[:-1][mask_time[1:]]
        _x0 = x[[0]][mask_time[[0]]]

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
        ell += CR @ y.sum(0)
        Lam += CRC * y.shape[0]
        V = la.inv(Lam)
        mu = V @ ell
        return mu, (V)

    def _forward(self,y,z,mask):
        mu = np.zeros((self.T,self.state_dim))
        V = np.zeros((self.T,self.state_dim,self.state_dim))
        m = self.m0(z[0]).copy()
        P = self.P0(z[0]).copy()
        for n in range(self.T):
            ''' update'''
            mu[n], V[n] = self.update(y[n,mask[n]],m,P,z=z[n])
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

    def sample_x(self,y,z,mask):
        mu, V = self._forward(y,z,mask)
        self._backward(mu,V,z)

    def loglikelihood(self,y,z=None,mask=None):
        z, mask = self._check_input(y=y,z=z,mask=mask)
        T,N = y.shape[:2]
        logl = np.zeros((T,N))
        for t in range(T):
            state = z[t]
            mu = self.C(state) @ self.x[t]
            Sigma = self.R(state)
            logl[t,mask[t]] = mvn_logpdf(y[t,mask[t]],mu,Sigma)
        return logl
        
    def _check_input(self,y,z=None,mask=None):
        if y.ndim != 3:
            raise ValueError("input must be 3d")
        if mask is None:
            mask = np.ones(y.shape[:2]).astype(bool)
        if mask.ndim != 2:
            raise ValueError("mask must be 2d")
        if y.shape[:2] != mask.shape[:2]:
            raise ValueError("mask must match y in dimensions")
        self.T, self.N, self.output_dim = y.shape

        if self.x.shape[0] != self.T:
            self._parameters['x'] = np.zeros((self.T,self.state_dim))

        if z is None:
            # Same state for all time.
            z = np.zeros(self.T).astype(int)
        if z.shape[0] != self.T:
            raise ValueError("1st dim of z and y must be equal.")
        return z, mask

    def forward(self,y,z=None,mask=None):
        # z is the state of the system at time t. 
        # mask is which data points are there
        z,mask = self._check_input(y=y,z=z,mask=mask)
        self.sample_x(y,z=z,mask=mask)
        if self.parameter_sampling == True:
            self.theta(y=y,x=self.x,labels=z,mask=mask)
        