from gibbs import get_mean, get_median, plot_cov_ellipse
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import gamma, wishart
import scipy.linalg as la
from typing import OrderedDict, Optional, Iterable
from tqdm import tqdm
import matplotlib.pyplot as plt


class Gibbs(object):
    r'''
    Gibbs sampler base class.

    Author: Julian Neri, May 2022
    '''
    def __init__(self):
        self._parameters = OrderedDict()
        self._samples = OrderedDict()
        self._estimates = OrderedDict()
        self._modules = OrderedDict()

    @property
    def nparams(self):
        return len(self._parameters)

    def __dir__(self) -> Iterable[str]:
        return list(self._parameters.keys())

    def __repr__(self) -> str:
        output = self.__class__.__name__  + " \n"
        for i in self._estimates.keys():
            output += " " + i +  " =  " + str(self._parameters[i]) + " \n"
        return output

    def add_module(self, name: str, module: Optional['object']) -> None:
        if not isinstance(module, object) and module is not None:
            raise TypeError("{} is not a object subclass".format(
                module))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\", got: {}".format(name))
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module

    def _append(self,params):
        for p in params:
            if p not in self._samples.keys():
                self._samples[p] = []
            self._samples[p].append(params[p].copy())

        for m in self._modules:
            self._modules[m]._append()

    def get_estimates(self,reduction='median',burn_rate=.75,skip_rate=1):
        if reduction == 'median':
            estim_fun = get_median
        else:
            estim_fun = get_mean

        chain = self.get_chain(burn_rate=burn_rate,skip_rate=skip_rate)
        for p in chain:
            self._estimates[p] = estim_fun(chain[p])

    def get_chain(self,burn_rate=0,skip_rate=1,flatten=False):
        chain = {}        
        skip_rate = int(max(skip_rate,1))
        for p in self._samples:
            num_samples = len(self._samples[p])
            burn_in = int(num_samples * burn_rate)
            stacked = np.stack(self._samples[p][burn_in::skip_rate],0)
            if flatten is True:
                stacked = stacked.reshape(stacked.shape[0],-1)   
            chain[p] = stacked.copy()
        return chain 

    def fit(self,data,model,samples=100,burn_rate=.75):
        for s in tqdm(range(samples)):
            _params = model(*data)
            self._append(_params)
        self.get_estimates(reduction='median',burn_rate=burn_rate)
    


class Module(object):
    r'''
    Gibbs module base class.

    Author: Julian Neri, May 2022
    '''
    def __init__(self):
        self._parameters = OrderedDict()

    @property
    def nparams(self):
        return len(self._parameters)

    # Used as a shortcut to get parameters : self._parameters['mu'] ==> self.mu
    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    # Used to ensure that parameters are set explicity, and not overwritten.
    def __setattr__(self, __name: str, __value) -> None:
        if '_parameters' in self.__dict__:
            if __name in self._parameters:
                raise AttributeError("Parameter '{}' must be set with '._parameters[key] = value'".format(__name))          
        self.__dict__[__name] = __value

    def __dir__(self) -> Iterable[str]:
        return list(self._parameters.keys())

    def __repr__(self) -> str:
        output = self.__class__.__name__  + " \n"
        for i in self._parameters.keys():
            output += " " + i +  " =  " + str(self._parameters[i]) + " \n"
        return output

class NormalWishart(Module):
    r'''
        Normal wishart system parameters

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,input_dim=1,hyper_sample=True,full_covariance=True):
        super(NormalWishart,self).__init__()
        self._dimo = output_dim
        self._dimi = input_dim
        self.hyper_sample = hyper_sample
        self.full_covariance = full_covariance

        self.initialize()

    def __call__(self, y, x):
        return self.forward(y,x)

    @property
    def output_dim(self):
        return self._dimo
    @output_dim.setter
    def output_dim(self,value):
        value = np.maximum(1,value)
        if value != self._dimo:
            self._dimo = value
            self.initialize()

    @property
    def input_dim(self):
        return self._dimi
    @input_dim.setter
    def input_dim(self,value):
        value = np.maximum(1,value)
        if value != self.dimi:
            self._dimi = value
            self.initialize()

    def initialize(self):
        self.define_priors()
        self.initialize_parameters()
        
    def define_priors(self):
        self.nu0 = self.output_dim + 1.0
        self.iW0 = np.eye(self.output_dim)

        self.a0 = self.nu0 / 2.0
        self.b0 = 1.0 / (2)

        self.c0 = .5
        self.d0 = .5
        
    def initialize_parameters(self):
        A = np.eye(self.output_dim,self.input_dim)
        A += np.random.normal(0,1e-3,A.shape)
        A /= np.abs(A).sum(-1)[:,None]
        Q = np.eye(self.output_dim)
        alpha = np.ones((self.input_dim))*1e-1

        self._parameters["A"] = A.copy()
        self._parameters["Q"] = Q.copy()
        self._parameters["alpha"] = alpha.copy()

    def _sample_A(self):
        tau = 1.0 / np.diag(self.Q)
        Alpha = np.diag(self.alpha)
        for row in range(self.output_dim):
            ell = tau[row] * (self.y[:,row] @ self.x)
            Lam = tau[row] *  ( (self.x.T @ self.x) + Alpha )
            Sigma = la.inv(Lam)
            mu = Sigma @ ell
            self._parameters['A'][row] = mvn.rvs(mu,Sigma)

    def _sample_alpha(self):
        if self.hyper_sample is False:
            return 0
        # The marginal prior over A is then a Cauchy distribution: N(a,|0,1/alpha)Gam(alpha|.5,.5) => Cauchy(a)
        a = np.zeros(self.input_dim)
        b = np.zeros(self.input_dim)
        for col in range(self.input_dim):
            a[col] = self.c0 + 1/2*self.output_dim
            b[col] = self.d0 + 1/2*( ((self.A[:,col]) ** 2.0)).sum(0)
        self._parameters['alpha'] = np.atleast_1d(gamma.rvs(a,scale=1.0/b))

    def _sample_Q(self):
        if self.full_covariance:
            Q = self._sample_cov_full()
        else:
            Q = self._sample_cov_diag()
        self._parameters['Q'] = Q

    def _sample_cov_diag(self):
        Nk = self.N + self.input_dim
        x_eps = self.y - self.x @ self.A.T
        quad = np.diag(x_eps.T @ x_eps) + np.diag((self.A) @ np.diag(self.alpha) @ (self.A).T)

        a_hat = self.a0 + 1/2 * Nk
        b_hat = self.b0 + 1/2 * quad

        tau = np.atleast_1d(gamma.rvs(a_hat,scale=1/b_hat))
        return np.diag(1/tau)

    def _sample_cov_full(self):
        nu = self.nu0 + self.N

        y_hat = (self.x @ self.A.T)
        x_eps = (self.y - y_hat)
        quad = x_eps.T @ x_eps
        iW = self.iW0 + quad
        iW = .5*(iW + iW.T)
        W = la.inv(iW)

        Lambda = np.atleast_2d(wishart.rvs(df=nu,scale=W) )
        return la.inv(Lambda)
        
    def forward(self,y,x):
        self.y = y
        self.x = x
        self.N = y.shape[0]
        self._sample_A()
        self._sample_Q()
        self._sample_alpha()
        return self._parameters




N = 500
x = np.ones((N,1))
mu = np.zeros(2)+10
Sigma = wishart.rvs(3,np.eye(2))
y = mvn.rvs(mu,Sigma,N)

model = NormalWishart(output_dim=2,input_dim=1,full_covariance=True)

sampler = Gibbs()
sampler.fit((y,x),model,1000)

chain = sampler.get_chain(flatten=True)

fig,ax = plt.subplots(len(chain),figsize=(5,len(chain)*1.5))
ax = np.atleast_1d(ax)
for j,s in enumerate(chain):
    ax[j].plot(chain[s],'k',alpha=.1,linewidth=1)
    ax[j].set_title(s)
ax[-1].set_xlabel('step')
plt.tight_layout()


plt.figure()
plt.scatter(y[:,0],y[:,1],s=10,alpha=.5)
mu_hat = sampler._estimates['A']
Sigma_hat = sampler._estimates['Q']
plot_cov_ellipse(pos=mu,cov=Sigma,nstd=1,fill=None,color='b',linewidth=2)
plot_cov_ellipse(pos=mu_hat,cov=Sigma_hat,nstd=1,fill=None,color='r',linewidth=2)
