#%%
from typing import OrderedDict, Optional, Iterable
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multinomial, gamma, beta
from tqdm import tqdm
import matplotlib.pyplot as plt
from .utils import get_mean,get_median

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

    def _sample(self,data):
        self._module.forward(data)
        
    def _fit_once(self,data):
        self._sample(data)
        self._append()

    def fit(self,data,samples=100,burn_rate=.75):
        for s in tqdm(range(samples)):
            self._fit_once(data)
        self._get_estimate(reduction='median',burn_rate=burn_rate)
    
    def __dir__(self) -> Iterable[str]:
        return list(self._parameters.keys())

    def __repr__(self) -> str:
        output = self.__class__.__name__  + " \n"
        for i in self._estimates.keys():
            output += " " + i +  " =  " + str(self._parameters[i]) + " \n"
        return output

    def plot_samples(self,params=None):
        k = self._samples.keys()
        if params in self._samples:
            k = list(params)
        fig,ax = plt.subplots(n,figsize=(5,len(k)*1.5))
        ax = np.atleast_1d(ax)
        for j,s in enumerate(k):
            stacked = np.stack(self._samples[s],0)
            ax[j].plot(stacked.reshape(stacked.shape[0],-1),'k',alpha=.1,linewidth=1)
            ax[j].set_title(s)
        ax[-1].set_xlabel('step')
        plt.tight_layout()
