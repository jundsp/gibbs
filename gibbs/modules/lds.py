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

from sequential.lds import ode_polynomial_predictor

#* Parameters should have a "sample /  learn" setting do register into the sampler. If not, then dont add to the chain, and allow for easy setting.

class StateSpace(Module):
    def __init__(self,output_dim=1,state_dim=2,hyper_sample=True,full_covariance=True,sigma_ev_sys=.1, sigma_ev_obs=.1, init_method="random"):
        super(StateSpace,self).__init__()
        self._dimy = output_dim
        self._dimx = state_dim
        self.hyper_sample = hyper_sample
        self.full_cov = full_covariance
        self.sigma_ev_sys = sigma_ev_sys
        self.sigma_ev_obs = sigma_ev_obs
        self.init_method = init_method
        self.initialize()
    
    def initialize(self):
        self.sys = NormalWishart(output_dim=self.state_dim, input_dim=self.state_dim,hyper_sample=self.hyper_sample,full_covariance=self.full_cov,sigma_ev=self.sigma_ev_sys,cov_sample=True)
        self.obs = NormalWishart(output_dim=self.output_dim, input_dim=self.state_dim,hyper_sample=self.hyper_sample,full_covariance=self.full_cov,sigma_ev=self.sigma_ev_obs)
        self.pri = NormalWishart(output_dim=self.state_dim, input_dim=1,hyper_sample=self.hyper_sample,full_covariance=self.full_cov,sigma_ev=1)
        self.pri._parameters['A'] *= 0
        self.I = np.eye(self.state_dim)

        if self.init_method == 'identity':
            self.sys._parameters['A'] = np.eye(self.state_dim)
            self.sys._parameters['Q'] = np.eye(self.state_dim)*self.sigma_ev_sys**2
            self.obs._parameters['A'] = np.eye(self.output_dim,self.state_dim)
            self.obs._parameters['Q'] = np.eye(self.output_dim)*self.sigma_ev_obs**2
        elif self.init_method == 'predict':
            self.sys._parameters['A'] = ode_polynomial_predictor(order=self.state_dim)
            self.sys._parameters['Q'] = np.eye(self.state_dim)*self.sigma_ev_sys**2
            self.obs._parameters['A'] = np.eye(self.output_dim,self.state_dim)
            self.obs._parameters['Q'] = np.eye(self.output_dim)*self.sigma_ev_obs**2
            

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

    def forward(self,y,x,x2,x1,x0):
        self.obs(y=y,x=x)
        self.sys(y=x2,x=x1)
        self.pri(y=x0)


class LDS(Module):
    r'''
        Bayesian linear dynamical system.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,state_dim=2,states=1,parameter_sampling=True,hyper_sample=True,full_covariance=True,init_method='random'):
        super(LDS,self).__init__()
        self._dimy = output_dim
        self._dimx = state_dim
        self._dimz = states
        self.parameter_sampling = parameter_sampling
        self.hyper_sample = hyper_sample
        self.full_cov = full_covariance
        self.init_method = init_method

        self._parameters["x"] = np.zeros((1,state_dim))
        self.initialize()

    def initialize(self):
        self.theta = TimePlate(*[StateSpace(output_dim=self.output_dim,state_dim=self.state_dim,hyper_sample=self.hyper_sample,full_covariance=self.full_cov,init_method=self.init_method) for i in range(self.states)])
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

    def generate(self,T=100):
        x = np.zeros((T,self.state_dim))
        x[0] = mvn.rvs(self.m0(0),self.P0(0))
        for t in range(1,T):
            x[t] = mvn.rvs(self.A(0) @ x[t-1], self.Q(0))
        y_clean = x @ self.C(0).T
        y = y_clean + mvn.rvs(np.zeros(self.output_dim),self.R(0),T)
        return y, y_clean, x
        
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
            ''' update '''
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

    def sample_parameters(self,data:'Data',z):
        for i,m in enumerate(self.theta):
            idx = z == i
            time_on = np.nonzero(idx)[0]
            _y, _x = [],[]
            for t in time_on:
                idx = data.time == t
                if idx.sum() > 0:
                    _y.append(data.output[idx])
                    _x.append(np.stack([self.x[t]]*idx.sum(),0))

            if len(_y) > 0:
                _y = np.concatenate(_y,0)
                _x = np.concatenate(_x,0)
            else:
                _y = data.output[[]]
                _x = self.x[[]]

            t2 = time_on[time_on>0]
            t1 = t2 - 1

            if len(t2) > 0:
                x2 = self.x[t2]
                x1 = self.x[t1]
            else:
                x2 = self.x[[]]
                x1 = self.x[[]]
            
            x0 = self.x[[0]]
            if 0 not in time_on:
                x0 = x0[np.array([False])] 

            m(y=_y,x=_x,x2=x2,x1=x1,x0=x0)

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
            self.sample_parameters(data,z=z)
        