import numpy as np
from .module import Module
from .plate import Plate
from .parameters import NormalWishart
from .mixture import Mixture, InfiniteMixture
from ..dataclass import Data
from ..distributions import Distribution, NormalWishart as NWdist, LaplaceGamma
from scipy.stats import multivariate_t as mvt
from scipy.special import logsumexp

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


class FiniteGMM(Module):
    r'''
        Finite Bayesian mixture of Gaussians.

        Collapsed Gibbs sampling. 

        Author: Julian Neri, 2023
    '''
    def __init__(self,components=4,output_dim=1,alpha=1,learn=False,full_covariance=True,sigma_ev:float=1):
        super().__init__()
        self._dimy = output_dim
        self.full_covariance = full_covariance
        self.sigma_ev = sigma_ev
        self.mix = InfiniteMixture(alpha=alpha,learn=learn)
        self._parameters['z'] = np.zeros(1,dtype=int)
        self.K = components

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
        nu0 = self.output_dim + 0.5
        k0 = .01

        s = nu0*self.sigma_ev**2.0
        S0 = np.eye(self.output_dim)*s
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

        pz = (Nk + self.mix.alpha / self.K) / (self.N + self.mix.alpha - 1)

        rho = 0.0
        if Nk > 0:
            rho = self._posterior_predictive(n,idx)
        else:
            rho = self._prior_predictive(n)
        return rho * pz

    def _sample_z_single(self,n):
        # Compute rho = p(z|y)
        rho = np.zeros(self.K)
        for k in range(self.K):
            rho[k] = self._get_rho_single(n,k)
        rho /= rho.sum()
        
        # Sample z
        _z_now = np.random.multinomial(1,rho).argmax()
        return _z_now
        
    def _sample_z(self):
        tau = np.random.permutation(self.N)
        for n in tau:
            self._parameters['z'][n] = self._sample_z_single(n)
        
    def _check_input(self,data: 'Data'):
        self.y = data.output
        self.N, self.output_dim = self.y.shape

        if self.z.shape[0] != self.y.shape[0]:
            self._parameters['z'] = np.zeros(self.N,dtype=int)-1

    def forward(self,data: 'Data'):
        self._check_input(data)
        self._sample_z()
        self.mix(self.z)


class InfiniteGMM(Module):
    r'''
        Infinite Bayesian mixture of Gaussians, a.k.a. Dirichlet process GMM.

        Collapsed Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,alpha=1,learn=True,hyper_sample=True,full_covariance=True,collapse_locally:bool=True,sigma_ev:float=1):
        super().__init__()
        self._dimy = output_dim
        self.hyper_sample = hyper_sample
        self.full_covariance = full_covariance
        self.sigma_ev = sigma_ev
        self.mix = InfiniteMixture(alpha=alpha,learn=learn)
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
        nu0 = self.output_dim + 0.5
        k0 = .01

        s = nu0*self.sigma_ev**2.0
        S0 = np.eye(self.output_dim)*s
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
                if np.all(self.z >= 0):
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


class InfiniteDistributionMix(InfiniteGMM):
    r'''
        Infinite Bayesian mixture of distributions, with different kinds of hyperparameters.

        Collapsed Gibbs sampling. 

        Author: Julian Neri, 2023
    '''
    def __init__(self, output_dim=1, alpha=1, learn=True, collapse_locally: bool = True, sigma_ev: float = 1, random_proposal:bool=False, sort_by_group:bool=False):
        self.kinds = []
        self.random_proposal = random_proposal
        self.sort_by_group = sort_by_group
        super().__init__(output_dim=output_dim,alpha=alpha,learn=learn,collapse_locally=collapse_locally,sigma_ev=sigma_ev)
        
    def initialize(self):
        self.dists = []
        self.dists.append(NWdist(sigma_ev=self.sigma_ev,output_dim=self.output_dim))
        self.dists.append(LaplaceGamma(sigma_ev=self.sigma_ev,output_dim=self.output_dim))

    @property
    def num_kinds(self):
        return len(self.dists)

    def _posterior_predictive(self, n, idx, dist:'Distribution'):
        return dist.posterior_predictive(self.data.output[n],self.data.output[idx])

    def _prior_predictive(self, n, dist:'Distribution'):
        return dist.prior_predictive(self.data.output[n])

    def _get_rho_single(self,n,k):
        idx = self.z == k
        idx[n] = False
        Nk = idx.sum()
        rho = -np.inf
        if Nk > 0:
            rho = self._posterior_predictive(n,idx,dist=self.dists[self.kinds[k]]) + np.log(Nk)
        return rho
    
    def _sample_z_single(self,n):
        # Compute rho = p(z|y)
        logrho_temp = np.zeros(self.num_kinds)-np.inf
        for i in range(self.num_kinds):
            logrho_temp[i] = self._prior_predictive(n,dist=self.dists[i])
        temp_sum = logsumexp(logrho_temp)
        logp_temp = logrho_temp - temp_sum
        if not np.isfinite(temp_sum):
            logp_temp = np.zeros_like(logrho_temp) - np.log(len(logrho_temp))

        kind_now = np.random.multinomial(1,np.exp(logp_temp)).argmax()
        if self.random_proposal:
            kind_now = np.random.multinomial(1,np.ones(self.num_kinds)/self.num_kinds).argmax()

        logrho = np.zeros(self.K+1)-np.inf
        logrho[-1] = logrho_temp[kind_now] + np.log(self.mix.alpha)
        for k in range(self.K):
            logrho[k] = self._get_rho_single(n,k)
        temp_sum = logsumexp(logrho)
        if not np.isfinite(temp_sum):
            logrho[:] = 0
            temp_sum = logsumexp(logrho)
        logrho -= temp_sum
        rho = np.exp(logrho)
        # Sample z
        _z_now = np.random.multinomial(1,rho).argmax()
        if _z_now > (self.K-1):
            _z_now = self.K + 0
            self.kinds.append(kind_now + 0)
            self.K += 1

        return _z_now
        
    def _collapse_groups(self):
        z_active = np.unique(self.z)
        self.K = len(z_active)
        temp = np.zeros_like(self.z)
        temp_kinds = []
        for k in range(self.K):
            idx = self.z == z_active[k]
            temp[idx] = k
            temp_kinds.append(self.kinds[z_active[k]])
        self._parameters['z'] = temp.copy()
        self.kinds = temp_kinds.copy()

    def _sample_z(self):
        if self.sort_by_group:
            tau = self.data.argsort_group()
        else:
            tau = np.random.permutation(self.N)
            
        for n in tau:
            self._parameters['z'][n] = self._sample_z_single(n)
            if self.collapse_locally:
                if np.all(self.z >= 0):
                    self._collapse_groups()

        if self.collapse_locally == False:
            self._collapse_groups()
        
    def _check_input(self, data: Data):
        self.data = data
        self.N = data.N
        self.output_dim = data.output_dim

        if self.z.shape[0] != self.N:
            self._parameters['z'] = np.zeros(self.N,dtype=int)-1
            self.K = 0
            self.kinds = []



class FiniteDistributionMix(FiniteGMM):
    r'''
        Finite Bayesian mixture of distributions, with flexible hyperpriors.

        Collapsed Gibbs sampling. 

        Author: Julian Neri, 2023
    '''
    def __init__(self, components=2, output_dim=1, alpha=1, learn=True, sigma_ev:float=1, random_proposal:bool=False, sort_by_group:bool=False):
        self.kinds = [0]*components
        self.kinds[0] = 1
        self.sort_by_group = sort_by_group
        super().__init__(components=components,output_dim=output_dim,alpha=alpha,learn=learn,sigma_ev=sigma_ev)
        
    def initialize(self):
        self.dists = []
        self.dists.append(NWdist(sigma_ev=self.sigma_ev,output_dim=self.output_dim))
        self.dists.append(LaplaceGamma(sigma_ev=self.sigma_ev,output_dim=self.output_dim))

    @property
    def num_kinds(self):
        return len(self.dists)

    def _posterior_predictive(self, n, idx, dist:'Distribution'):
        return dist.posterior_predictive(self.data.output[n],self.data.output[idx])

    def _prior_predictive(self, n, dist:'Distribution'):
        return dist.prior_predictive(self.data.output[n])

    def _get_rho_single(self,n,k):
        idx = self.z == k
        idx[n] = False
        Nk = idx.sum() 

        pz = (Nk + self.mix.alpha / self.K) / (self.N + self.mix.alpha - 1)

        rho = -np.inf
        if Nk > 0:
            rho = self._posterior_predictive(n,idx,dist=self.dists[self.kinds[k]])
        else:
            rho = self._prior_predictive(n,dist=self.dists[self.kinds[k]])
        return rho + np.log(pz)
    

    def _sample_z_single(self,n):
        # Compute rho = p(z|y)
        logrho = np.zeros(self.K) - np.inf
        for k in range(self.K):
            logrho[k] = self._get_rho_single(n,k)

        logpy = logsumexp(logrho)
        if not np.isfinite(logpy):
            logrho[:] = 0
            logpy = logsumexp(logrho)
        logrho -= logpy
        rho = np.exp(logrho)
        
        # Sample z
        _z_now = np.random.multinomial(1,rho).argmax()
        return _z_now
    

    def _sample_z(self):
        if self.sort_by_group:
            tau = self.data.argsort_group()
        else:
            tau = np.random.permutation(self.N)
            
        for n in tau:
            self._parameters['z'][n] = self._sample_z_single(n)

    def _check_input(self,data: 'Data'):
        self.data = data
        self.N = data.N
        self.output_dim = data.output_dim

        if self.z.shape[0] != self.N:
            self._parameters['z'] = np.zeros(self.N,dtype=int)-1
