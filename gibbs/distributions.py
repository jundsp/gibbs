import numpy as np
import scipy.stats as stats
from scipy.special import gamma, gammaln
from .kernels import RBF
import scipy.linalg as la
from .utils import gamma_moments2params, mvt_logpdf
from typing import List

class Distribution(object):
    def __init__(self) -> None:
        pass

    def posterior(self,y:np.ndarray,x:np.ndarray=None,x_star:np.ndarray=None):
        return 0
    
    def predictive(self,y:np.ndarray,a,b):
        return 0
    
    def predictive_parameters(self):
        return 0
    
    def posterior_predictive(self,y_star:np.ndarray,y:np.ndarray,x_star:np.ndarray=None,x:np.ndarray=None):
        return 0

    def prior_predictive(self,y_star:np.ndarray,x_star:np.ndarray=None):
        return 0
    

class LaplaceGamma(Distribution):
    def __init__(self,sigma_ev=1,output_dim=1) -> None:

        self.output_dim = output_dim
        sigma_ev = np.asarray(sigma_ev).ravel().astype(float)
        self.a0 = 2
        self.b0 = self.a0 * sigma_ev ** 2

    def posterior(self,y,x=None,x_star=None):
        y = np.atleast_2d(y)
        a_hat = self.a0 + y.shape[0]
        b_hat = self.b0 + np.abs(y).sum(0)
        return a_hat, b_hat
    
    def predictive(self,y,a,b,reduce=True):
        y = np.atleast_2d(y)
        b = np.atleast_2d(b)
        a_hat = a + 1
        b_hat = b + np.abs(y)
        logp = -1*np.log(2) + a * np.log(b) - a_hat*np.log(b_hat) + gammaln(a_hat) - gammaln(a)
        if reduce:
            logp = logp.sum()
        return logp
    
    def predictive_parameters(self,a,b):
        return a, b
    
    def params2moments(self,a, b):
        mean = np.zeros((len(b),1))
        var = (2*b**2)/(2 - 3*a + a**2)
        cov = np.diag(var)
        return mean, cov
    
    def posterior_predictive(self,y_star,y,x_star=None,x=None,reduce=True):
        a, b, = self.posterior(y)
        return self.predictive(y_star,a=a,b=b,reduce=reduce)

    def prior_predictive(self,y_star,x_star=None,reduce=True):
        return self.predictive(y_star,a=self.a0,b=self.b0,reduce=reduce)
    

class Cauchy(Distribution):
    def __init__(self,scale=1,output_dim=1) -> None:

        self.output_dim = output_dim
        scale = np.asarray(scale).ravel().astype(float)
        self.scale = scale

    def posterior(self,y=None,x=None,x_star=None):
        return np.zeros_like(self.scale), self.scale
    
    def predictive(self,y,scale):
        y = np.atleast_2d(y)
        scale = np.atleast_2d(scale)
        p = stats.cauchy.logpdf(x=y,loc=0,scale=scale)
        return p
    
    def predictive_parameters(self,loc,scale):
        return loc, scale
    
    def params2moments(self,loc, scale):
        return loc, scale
    
    def posterior_predictive(self,y_star,y):
        return self.predictive(y_star,scale=self.scale)

    def prior_predictive(self,y_star):
        return self.predictive(y_star,scale=self.scale)
    

class NormalWishart(Distribution):
    def __init__(self,sigma_ev=1,output_dim=1) -> None:

        self.output_dim = output_dim
        m0 = np.zeros(output_dim)
        nu0 = self.output_dim + .5
        k0 = .01

        s = nu0*sigma_ev**2.0
        S0 = np.eye(output_dim)*s

        self.m0 = m0
        self.k0 = k0
        self.S0 = S0
        self.nu0 = nu0

    def posterior(self,y,x=None,x_star=None):
        N = y.shape[0]
        yk = np.atleast_2d(y)
        y_bar = yk.mean(0)
        S = yk.T @ yk
        kN = self.k0 + N
        mN = (self.k0 * self.m0 + y_bar * N)/kN
        nuN = self.nu0 + N
        SN = self.S0 + S + self.k0*np.outer(self.m0,self.m0) - kN*np.outer(mN,mN)
        return mN,kN,SN,nuN
    
    def predictive_parameters(self,m,k,S,nu):
        _m = m
        _nu = nu - self.output_dim + 1.0
        _S = (k + 1.0)/(k*_nu) * S
        return _m, _S, _nu
    
    def params2moments(self,m,S,nu):
        cov = S * (nu)/(nu-2)
        return m, cov
    
    def predictive(self,y,m,k,S,nu):
        _m, _S, _nu = self.predictive_parameters(m,k,S,nu)
        logp = stats.multivariate_t.logpdf(y,loc=_m,shape=_S,df=_nu)
        return logp.sum()

    def posterior_predictive(self,y_star,y,x_star=None,x=None):
        m,k,S,nu = self.posterior(y)
        return self.predictive(y_star,m,k,S,nu)

    def prior_predictive(self,y_star,x_star=None):
        return self.predictive(y_star,self.m0,self.k0,self.S0,self.nu0)
    
class GaussianProcess(Distribution):
    def __init__(self,beta=1,sigma_ev=1,var_factor=1,output_dim=1,kernel:'RBF'=None) -> None:

        if kernel is None:
            kernel = RBF()

        self.output_dim = output_dim
        sigma_ev = np.asarray(sigma_ev).ravel().astype(float)
        self.dim = len(sigma_ev)

        precision_ev = 1/sigma_ev**2
        precision_var = (var_factor*precision_ev)**2
        self.a0, self.b0 = gamma_moments2params(mu=precision_ev,var=precision_var)

        self.beta = beta
        self.kernel = kernel
    
    def posterior(self,x_star,y,x):
        if x_star.ndim != 2:
            raise ValueError("x_star must be 2d")
        if y.ndim != 2:
            raise ValueError("y must be 2d")
        if x.ndim != 2:
            raise ValueError("x must be 2d")
        
        K = self.kernel(x,x)
        I = np.eye(K.shape[0])
        C = K + I / self.beta
        Ci = la.inv(C)
        ell = Ci @ y

        _N = len(x)
        a_hat = self.a0 + 1/2*_N
        b_hat = self.b0 + 1/2*(y.T @ Ci @ y).ravel()

        output_dim = y.shape[-1]
        N_star = x_star.shape[0]
        m, Sigma = np.zeros((N_star,output_dim)), np.zeros((N_star,output_dim,output_dim))

        # batch this?
        for n in range(N_star):
            k = self.kernel(x,x_star[[n]])
            c = self.kernel(x_star[[n]],x_star[[n]]) + 1/self.beta

            m[n] = k.T @ ell
            Sigma[n] = c - k.T @ Ci @ k  
        return m, Sigma, a_hat, b_hat
    
    def predictive_parameters(self,m,sigma2,a_hat,b_hat):
        var = b_hat/a_hat * sigma2
        nu = 2*a_hat
        return m, var, nu
    
    def params2moments(self,m,var,nu):
        cov = var * (nu)/(nu-2)
        return m, cov
    
    def predictive(self,y,m,var,nu,reduce=True):
        # N = y.shape[0]   
        # logp = np.zeros(N)
        # for n in range(N):
        #     logp[n] = stats.multivariate_t.logpdf(y[n],loc=m[n],shape=var[n],df=nu)
             
        logp = mvt_logpdf(y=y,loc=m,shape=var,df=nu)
        if reduce:
            logp = logp.sum()
        return logp
        
    def posterior_predictive(self,y_star,x_star,y,x,reduce=True):
        if x_star.ndim != 2:
            raise ValueError("x_star must be 2d")
        if y_star.ndim != 2:
            raise ValueError("y_star must be 2d")
        if y.ndim != 2:
            raise ValueError("y must be 2d")
        if x.ndim != 2:
            raise ValueError("x must be 2d")
        m, Sigma, a_hat, b_hat = self.posterior(x_star,y,x)     
        m, var, nu = self.predictive_parameters(m,Sigma,a_hat,b_hat)
        return self.predictive(y_star,m=m,var=var,nu=nu,reduce=reduce)

    def prior_predictive(self,y_star,x_star,reduce=True):
        if x_star.ndim != 2:
            raise ValueError("x_star must be 2d")
        if y_star.ndim != 2:
            raise ValueError("y_star must be 2d")
        output_dim = y_star.shape[-1]
        N_star = x_star.shape[0]
        m, Sigma = np.zeros((N_star,output_dim)), np.zeros((N_star,output_dim,output_dim))
        for n in range(N_star):
            Sigma[n] = self.kernel(x_star[[n]],x_star[[n]]) + 1/self.beta  
        m, var, nu = self.predictive_parameters(m,Sigma,self.a0,self.b0)
        return self.predictive(y_star,m=m,var=var,nu=nu,reduce=reduce)
    

class MultidimDistribution(Distribution):
    def __init__(self,dists:List[Distribution]) -> None:
        super().__init__()
        self.list_of_dists = dists

    def posterior_predictive(self, y_star: np.ndarray, y: np.ndarray, x_star: np.ndarray = None, x: np.ndarray = None, reduce=True):
        p = 0
        for i in range(y_star.shape[-1]):
            p += self.list_of_dists[i].posterior_predictive(y_star=y_star[:,[i]],y=y[:,[i]],x_star=x_star,x=x,reduce=reduce)
        return p
    
    def prior_predictive(self, y_star: np.ndarray, x_star: np.ndarray = None, reduce=True):
        p = 0
        for i in range(y_star.shape[-1]):
            p += self.list_of_dists[i].prior_predictive(y_star=y_star[:,[i]],x_star=x_star,reduce=reduce)
        return p