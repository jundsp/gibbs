#%%
from typing import OrderedDict, Optional, Iterable
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multinomial, gamma, beta
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

    # def __getattr__(self, name: str) -> Any:
    #     if '_parameters' in self.__dict__:
    #         _parameters = self.__dict__['_parameters']
    #         if name in _parameters:
    #             return _parameters[name]
    #     raise AttributeError("'{}' object has no attribute '{}'".format(
    #         type(self).__name__, name))

    # def __setattr__(self, __name: str, __value: Any) -> None:
    #     if '_parameters' in self.__dict__:
    #         if __name in self._parameters:
    #             raise AttributeError("Parameter '{}' must be set with '._parameters[key] = value'".format(__name))
    #     self.__dict__[__name] = __value

    def _append(self):
        for p in self._parameters:
            if p not in self._samples.keys():
                self._samples[p] = []
            self._samples[p].append(self._parameters[p].copy())

    def _get_median(self,burn_rate=.75,skip_sample=1):
        for p in self._samples:
            num_samples = len(self._samples[p])
            burn_in = int(num_samples * burn_rate)
            stacked = np.stack(self._samples[p][burn_in::skip_sample],0)
            self._estimates[p] = np.median(stacked,0)

    def _sample(self):
        for p in self._parameters:
            _sampler_fn =getattr(self,"_sample_"+p,None)
            if callable(_sampler_fn):
                _sampler_fn()
            else:
                print("Sampler for parameter '{}' is not implemented.".format(p))
        

    def fit(self,samples=100,burn_rate=.5):
        for s in tqdm(range(samples)):
            self._sample()
            self._append()
        self._get_median(burn_rate=burn_rate)

    def __dir__(self) -> Iterable[str]:
        return list(self._parameters.keys())

    def __repr__(self) -> str:
        output = "Gibbs \n"
        for i in self._estimates.keys():
            output += " " + i +  " =  " + str(self._parameters[i]) + " \n"
        return output

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
            pi_ = np.array([pi_eta,1-pi_eta])
            pi_ = pi_.ravel()
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
        z_unique = np.unique(self._parameters['z'])
        self.K = len(z_unique)
        temp = self._parameters['z'].copy()
        for k in range(self.K):
            idx = self._parameters['z'] == z_unique[k]
            temp[idx] = k
        self._parameters['z'] = temp.copy()
        self._hyperparameters = [self._hyperparameters[k] for k in z_unique]
        self._kinds = [self._kinds[k] for k in z_unique]

    def plot_samples(self):
        n = len(self._samples)
        fig,ax = plt.subplots(n,figsize=(5,n*1.5))
        for j,s in enumerate(self._samples):
            stacked = np.stack(self._samples[s],0)
            ax[j].plot(stacked,'k',alpha=.5)
            ax[j].set_title(s)
        plt.tight_layout()


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