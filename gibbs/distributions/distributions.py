import numpy as np
import scipy.stats as stats
from scipy.special import gamma, gammaln

class Distribution(object):
    def __init__(self) -> None:
        pass

    def posterior(self,y):
        return 0
    
    def predictive(self,y,a,b):
        return 0
    
    def predictive_parameters(self):
        return 0
    
    def posterior_predictive(self,y_star,y):
        return 0

    def prior_predictive(self,y_star):
        return 0
    

class LaplaceGamma(Distribution):
    def __init__(self,sigma_ev=1,output_dim=1) -> None:

        self.output_dim = output_dim
        sigma_ev = np.asarray(sigma_ev).ravel().astype(float)
        self.dim = len(sigma_ev)
        self.a0 = .5
        self.b0 = self.a0 * sigma_ev ** 2

    def posterior(self,y):
        y = np.atleast_2d(y)
        a_hat = self.a0 + y.shape[0]
        b_hat = self.b0 + np.abs(y).sum(0)
        return a_hat, b_hat
    
    def predictive(self,y,a,b):
        y = np.atleast_2d(y)
        b = np.atleast_2d(b)
        a_hat = a + 1
        b_hat = b + np.abs(y)
        logp = -1*np.log(2) + a * np.log(b) - a_hat*np.log(b_hat) + gammaln(a_hat) - gammaln(a)
        return np.exp(logp.sum(-1))
    
    def predictive_parameters(self,a,b):
        return a, b
    
    def params2moments(self,a, b):
        mean = np.zeros_like(b)
        var = (2*b**2)/(2 - 3*a + a**2)
        cov = np.diag(var)
        return mean, cov
    
    def posterior_predictive(self,y_star,y):
        a, b, = self.posterior(y)
        return self.predictive(y_star,a=a,b=b)

    def prior_predictive(self,y_star):
        return self.predictive(y_star,a=self.a0,b=self.b0)
    

class NormalWishart(Distribution):
    def __init__(self,sigma_ev=1,output_dim=1) -> None:

        self.output_dim = output_dim
        m0 = np.zeros(output_dim)
        nu0 = self.output_dim + 0.5
        k0 = .01

        s = nu0*sigma_ev**2.0
        S0 = np.eye(output_dim)*s

        self.m0 = m0
        self.k0 = k0
        self.S0 = S0
        self.nu0 = nu0

    def posterior(self,y):
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
        return stats.multivariate_t.pdf(y,loc=_m,shape=_S,df=_nu)

    def posterior_predictive(self,y_star,y):
        m,k,S,nu = self.posterior(y)
        return self.predictive(y_star,m,k,S,nu)

    def prior_predictive(self,y_star):
        return self.predictive(y_star,self.m0,self.k0,self.S0,self.nu0)