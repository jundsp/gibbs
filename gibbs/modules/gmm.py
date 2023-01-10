import numpy as np
from .module import Module
from .plate import Plate
from .parameters import NormalWishart
from .mixture import Mixture, InfiniteMixture
from ..dataclass import Data
from scipy.stats import multivariate_t as mvt

class GMM(Module):
    r'''
        Finite Bayesian mixture of Gaussians.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,components=3,hyper_sample=True,full_covariance=True):
        super().__init__()
        self._dimy = output_dim
        self._dimz = components
        self.hyper_sample = hyper_sample
        self.full_covariance = full_covariance
        self.initialize()

    def initialize(self):
        self.theta = Plate(*[NormalWishart(output_dim=self.output_dim,hyper_sample=self.hyper_sample,full_covariance=self.full_covariance) for i in range(self.components)])
        self.mix = Mixture(components=self.components)

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
    def components(self):
        return self._dimz
    @components.setter
    def components(self,value):
        value = np.maximum(1,value)
        if value != self._dimz:
            self._dimz = value
            self.initialize()

    def loglikelihood(self,data: 'Data'):
        loglike = np.zeros((len(data),self.components))
        for k in range(self.components):
            loglike[:,k] = self.theta[k].loglikelihood(data.output)
        return loglike

    def logjoint(self,y,mask=None):
        mask = self._check_input(y,mask)
        rz,cz = np.nonzero(mask)
        logl = self.loglikelihood(y,mask)
        temp = logl[rz,cz]
        temp = temp[np.arange(temp.shape[0]),self.mix.z]
        return temp

    def forward(self,data: 'Data'):
        loglike = self.loglikelihood(data)
        self.mix(loglike)
        self.theta(data.output,self.mix.z)


class InfiniteGMM(Module):
    r'''
        Infinite Bayesian mixture of Gaussians.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,alpha=1,learn=True,hyper_sample=True,full_covariance=True,collapse_locally:bool=True):
        super().__init__()
        self._dimy = output_dim
        self.hyper_sample = hyper_sample
        self.full_covariance = full_covariance
        self.learn = learn
        self.alpha = alpha
        self.mix = InfiniteMixture(alpha=self.alpha,learn=self.learn)
        self._parameters['z'] = np.zeros(1,dtype=int) -1
        self.K = 0
        self.collapse_locally = collapse_locally

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

    def initialize(self):
        m0 = np.zeros(self.output_dim)
        nu0 = self.output_dim+1
        k0 = .01
        S0 = np.eye(self.output_dim)*nu0*.1
        self.theta = tuple([m0,k0,S0,nu0])

    def _posterior(self,y,m0,k0,S0,nu0):
        N = y.shape[0]
        yk = np.atleast_2d(y)
        y_bar = yk.mean(0)
        S = yk.T @ yk
        kN = k0 + N
        mN = (k0 * m0 + y_bar * N)/kN
        nuN = nu0 + N
        SN = S0 + S + k0*np.outer(m0,m0) - kN*np.outer(mN,mN)
        return mN,kN,SN,nuN

    def _posterior_predictive(self,n,idx):
        m0,k0,S0,nu0 = self.theta
        m,k,S,nu = self._posterior(self.y[idx],m0,k0,S0,nu0)
        return self._predictive(self.y[n],m,k,S,nu)

    def _prior_predictive(self,n):
        m0,k0,S0,nu0 = self.theta
        return self._predictive(self.y[n],m0,k0,S0,nu0)

    def _predictive_parameters(self,m,k,S,nu):
        _m = m
        _nu = nu - self.output_dim + 1.0
        _S = (k + 1.0)/(k*_nu) * S
        return _m, _S, _nu

    def _predictive(self,y,m,k,S,nu):
        _m, _S, _nu = self._predictive_parameters(m,k,S,nu)
        return mvt.pdf(y,loc=_m,shape=_S,df=_nu)

    def _get_rho_single(self,n,k):
        idx = self.z == k
        idx[n] = False
        Nk = idx.sum() 

        rho = 0.0
        if Nk > 0:
            rho = self._posterior_predictive(n,idx) * Nk
        return rho

    def _sample_z_single(self,n):
        # Compute rho = p(z|y)
        rho = np.zeros(self.K + 1)
        rho[-1] = self._prior_predictive(n) * self.mix.alpha
        for k in range(self.K):
            rho[k] = self._get_rho_single(n,k)
        rho /= rho.sum()

        # Sample z
        _z_now = np.random.multinomial(1,rho).argmax()
        if _z_now > (self.K-1):
            self.K += 1

        return _z_now
        
    def _sample_z(self):
        tau = np.random.permutation(self.N)
        for n in tau:
            self._parameters['z'][n] = self._sample_z_single(n)
            if self.collapse_locally:
                self._collapse_groups()

        if self.collapse_locally == False:
            self._collapse_groups()

    def _collapse_groups(self):
        z_active = np.unique(self.z)
        self.K = len(z_active)
        temp = np.zeros_like(self.z)
        for k in range(self.K):
            idx = self.z == z_active[k]
            temp[idx] = k
        self._parameters['z'] = temp.copy()
        
    def _check_input(self,data: 'Data'):
        self.y = data.output
        self.N, self.output_dim = self.y.shape

        if self.z.shape[0] != self.y.shape[0]:
            self._parameters['z'] = np.zeros(self.N,dtype=int)-1
            self.K = 0

    def forward(self,data: 'Data'):
        self._check_input(data)
        self._sample_z()
        self.mix(self.z)