from typing import OrderedDict, Optional, Iterable, overload, Set
from itertools import islice
import operator
import numpy as np
from tqdm import tqdm

from scipy.stats import multivariate_normal as mvn
from scipy.stats import gamma, wishart, dirichlet
from scipy.special import logsumexp
import scipy.linalg as la

from .utils import get_mean,get_median, mvn_logpdf
from .dataclass import Data
from .modules.module import Module

class Gibbs(object):
    r'''
    Gibbs sampler base class.

    Author: Julian Neri, May 2022
    '''
    def __init__(self):
        self._samples = OrderedDict()
        self._estimates = OrderedDict()
        self.step_count = 0

    @property
    def nparams(self):
        return len(self._estimates)

    def __dir__(self) -> Iterable[str]:
        return list(self._estimates.keys())

    def __repr__(self) -> str:
        output = self.__class__.__name__  + " \n"
        for i in self._estimates.keys():
            output += " " + i +  " =  " + str(self._estimates[i]) + " \n"
        return output

    def __len__(self) -> int:
        return self.step_count

    def __call__(self, params):
        return self.step(params)

    def get_estimates(self,reduction='median',burn_rate=.75,skip_rate=1):
        if reduction == 'median':
            estim_fun = get_median
        else:
            estim_fun = get_mean

        chain = self.get_chain(burn_rate=burn_rate,skip_rate=skip_rate)
        for p in chain:
            self._estimates[p] = estim_fun(chain[p])

    def get_chain(self,burn_rate:float=0,skip_rate=1,flatten=False):
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

    def step(self,params):
        for name,value in params:
            if name not in self._samples.keys():
                self._samples[name] = []
            self._samples[name].append(value.copy())
        self.step_count += 1

    def fit(self,data: 'Data', model: 'Module',samples=10):
        for iter in tqdm(range(samples)):
            model(data)
            self.step(model.named_parameters())
        self.get_estimates()