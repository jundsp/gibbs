import numpy as np
from .module import Module
from .plate import Plate
from .parameters import NormalWishart
from .mixture import Mixture

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

    def loglikelihood(self,y,mask):
        loglike = np.zeros((self.T,self.N,self.components))
        rz, cz = np.nonzero(mask)
        for k in range(self.components):
            loglike[rz,cz,k] = self.theta[k].loglikelihood(y[rz,cz])
        return loglike

    def logjoint(self,y,mask=None):
        mask = self._check_input(y,mask)
        rz,cz = np.nonzero(mask)
        logl = self.loglikelihood(y,mask)
        temp = logl[rz,cz]
        temp = temp[np.arange(temp.shape[0]),self.mix.z]
        return temp

    def _check_input(self,y,mask=None):
        if y.ndim != 3:
            raise ValueError("input must be 3d")
        if mask is None:
            mask = np.ones(y.shape[:2]).astype(bool)
        if mask.ndim != 2:
            raise ValueError("mask must be 2d")
        if y.shape[:2] != mask.shape[:2]:
            raise ValueError("mask must match y in dimensions")
        self.T, self.N, self.output_dim = y.shape
        return mask

    def forward(self,y,mask=None):
        mask = self._check_input(y,mask)
        loglike = self.loglikelihood(y,mask)
        rz, cz = np.nonzero(mask)
        self.mix(loglike[rz,cz])
        self.theta(y[rz,cz],self.mix.z)

