#%%
from typing import OrderedDict, Optional, Iterable
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multinomial, gamma, beta
from tqdm import tqdm
import matplotlib.pyplot as plt
from .utils import get_mean,get_median

import torch.nn.modules.module

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

    def register_parameter(self, name: str, param: Optional[np.ndarray]) -> None:
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter): parameter to be added to the sampler.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Gibbs.__init__() call")

        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        else:
            self._parameters[name] = param

    def add_module(self, name: str, module: Optional['Gibbs']) -> None:
        if not isinstance(module, Gibbs) and module is not None:
            raise TypeError("{} is not a Gibbs subclass".format(
                module))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\", got: {}".format(name))
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module

    # Used as a shortcut to get parameters : self._parameters['mu'] ==> self.mu
    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_modules' in self.__dict__:
            _modules = self.__dict__['_modules']
            if name in _modules:
                return _modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    # Used to ensure that parameters are set explicity, and not overwritten.
    def __setattr__(self, __name: str, __value) -> None:
        if '_parameters' in self.__dict__:
            if __name in self._parameters:
                raise AttributeError("Parameter '{}' must be set with '._parameters[key] = value'".format(__name))
         
        modules = self.__dict__.get('_modules')
        if isinstance(__value, Gibbs):
            if modules is None:
                raise AttributeError(
                    "cannot assign module before Gibbs.__init__() call")
            modules[__name] = __value
        elif modules is not None and __name in modules:
            if __value is not None:
                raise TypeError("cannot assign '{}' as child module '{}' "
                                "(Gibbs or None expected)"
                                .format(torch.typename(__value), __name))
            modules[__name] = __value            
        else:
            self.__dict__[__name] = __value

    def _append(self):
        for p in self._parameters:
            if p not in self._samples.keys():
                self._samples[p] = []
            self._samples[p].append(self._parameters[p].copy())

        for m in self._modules:
            self._modules[m]._append()

    def _get_estimate(self,reduction='median',burn_rate=.75,skip_rate=1):
        if reduction == 'median':
            estim_fun = get_median
        else:
            estim_fun = get_mean
            
        skip_rate = int(max(skip_rate,1))

        for p in self._samples:
            num_samples = len(self._samples[p])
            burn_in = int(num_samples * burn_rate)
            stacked = np.stack(self._samples[p][burn_in::skip_rate],0)
            self._estimates[p] = estim_fun(stacked)

    def get_chain(self,burn_rate=0,skip_rate=1,flatten=False):
        chain = {}        
        skip_rate = int(max(skip_rate,1))
        for p in self._samples:
            num_samples = len(self._samples[p])
            burn_in = int(num_samples * burn_rate)
            stacked = np.stack(self._samples[p][burn_in::skip_rate],0)
            if flatten is True:
                stacked = stacked.ravel()    
            chain[p] = stacked.copy()
        return chain 

    def _sample(self):
        for p in self._parameters:
            _sampler_fn =getattr(self,"_sample_"+p,None)
            if callable(_sampler_fn):
                _sampler_fn()
            else:
                print("Sampler for parameter '{}' is not implemented.".format(p))

        for m in self._modules:
            _updater_fn =getattr(self,"_update_"+m,None)
            if callable(_updater_fn):
                _updater_fn()
                self._modules[m]._sample()
            else:
                print("Updater for module '{}' is not implemented.".format(m))
        
    def _fit_once(self):
        self._sample()
        self._append()

    def fit(self,samples=100,burn_rate=.75):
        for s in tqdm(range(samples)):
            self._fit_once()
        self._get_estimate(reduction='median',burn_rate=burn_rate)
    
    def __dir__(self) -> Iterable[str]:
        return list(self._parameters.keys())

    def __repr__(self) -> str:
        output = self.__class__.__name__  + " \n"
        for i in self._estimates.keys():
            output += " " + i +  " =  " + str(self._parameters[i]) + " \n"
        return output

    def plot_samples(self,params=None):
        if params in self._samples:
            k = list(params)
        else:
            k = self._samples.keys()
        n = len(k)
        fig,ax = plt.subplots(n,figsize=(5,n*1.5))
        ax = np.atleast_1d(ax)
        for j,s in enumerate(k):
            stacked = np.stack(self._samples[s],0)
            if s == 'z':
                ax[j].plot(stacked.transpose(1,0),'b',alpha=.05)
                ax[j].plot(self._estimates['z'],'k')
            else:
                ax[j].plot(stacked.reshape(stacked.shape[0],-1),'k',alpha=.1,linewidth=1)
            ax[j].set_title(s)
            # ax[j].set_ylim(0)
        ax[-1].set_xlabel('step')
        plt.tight_layout()

class GibbsDirichletProcess(Gibbs):
    '''
    Gibbs sampler sub-class for the Dirichlet process.

    Author: Julian Neri, May 2022
    '''
    def __init__(self,alpha=1,learn=True):
        super().__init__()
        self.K = 0
        self._learn = learn
        self.register_parameter("z",None)
        self.register_parameter("eta",np.array([.5]))
        self.register_parameter("alpha",np.atleast_1d(alpha))

    def _sample_alpha(self):
        if self._learn:
            K = len(np.unique(self._parameters['z']))
            a = 1
            b = 1
            b_hat = b - np.log(self._parameters['eta'])
            y = a + K - 1
            z = self.N * b_hat
            pi_eta = y/(y+z)
            pi_ = np.array([pi_eta,1-pi_eta]).ravel()
            m = multinomial.rvs(1.0,pi_).argmax()
            if m == 0:
                a_hat = a + K
            else:
                a_hat = a + K - 1
            self._parameters['alpha'] = gamma.rvs(a=a_hat,scale=1/b_hat)

    def _sample_eta(self):
        if self._learn:
            a = self._parameters['alpha'] + 1.0
            b = self.N
            self._parameters['eta'] = beta.rvs(a,b)

    def _collapse_groups(self):
        z_active = np.unique(self._parameters['z'])
        self.K = len(z_active)
        temp = self._parameters['z'].copy()
        for k in range(self.K):
            idx = self._parameters['z'] == z_active[k]
            temp[idx] = k
        self._parameters['z'] = temp.copy()
        
        if "_hyperparameters" in self.__dict__:
            if len(self._hyperparameters) > 0:
                self._hyperparameters = [self._hyperparameters[k] for k in z_active]
        
        if "_kinds" in self.__dict__:
            if len(self._kinds) > 0:
                self._kinds = [self._kinds[k] for k in z_active]


class DirichletProcess(object):
    '''
    Dirichlet process model, for generation.

    Author: Julian Neri, 2022
    '''
    def __init__(self,alpha=1):
        self.alpha = alpha
        self.reset()
        
    def reset(self):
        self.Nk = []
        self.N = 0
        self.K = 0

    def sample(self):
        if self.K == 0:
            self.Nk = [1]
            self.K = 1
            z = 0
        else:
            pk = np.zeros(self.K+1)
            Z = self.N+self.alpha-1
            for k in range(self.K):
                pk[k] = self.Nk[k] / Z
            pk[-1] = self.alpha / Z
            pk /= pk.sum()
            z = np.random.multinomial(1,pk).argmax(-1)
            if z > self.K - 1:
                self.Nk.append(1)
                self.K += 1
            else:
                self.Nk[z] += 1
        self.N += 1
        return z