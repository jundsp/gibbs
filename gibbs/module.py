from typing import OrderedDict,overload,Optional, Iterable, Set
from itertools import islice
import operator
import numpy as np

from scipy.stats import multivariate_normal as mvn
from scipy.stats import gamma, wishart, dirichlet
from scipy.special import logsumexp
import scipy.linalg as la

from .utils import mvn_logpdf

class Module(object):
    r'''
    Gibbs module base class.

    Author: Julian Neri, May 2022
    '''
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    @property
    def nparams(self):
        return len(self._parameters)

    def add_module(self, name: str, module: Optional['Module']) -> None:
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
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
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    # Used to ensure that parameters are set explicity, and not overwritten.
    def __setattr__(self, __name: str, __value) -> None:
        if '_parameters' in self.__dict__:
            if __name in self._parameters:
                raise AttributeError("Parameter '{}' must be set with '._parameters[key] = value'".format(__name))

        modules = self.__dict__.get('_modules')
        if isinstance(__value, Module):
            if modules is None:
                raise AttributeError(
                    "cannot assign module before Module.__init__() call")
            modules[__name] = __value
        elif modules is not None and __name in modules:
            if __value is not None:
                raise TypeError("cannot assign as child module (Module or None expected)")
            modules[__name] = __value
        else:
            super().__setattr__(__name,__value)

    def __dir__(self) -> Iterable[str]:
        return list(self._parameters.keys())

    def _get_name(self):
        return self.__class__.__name__

    def __repr__(self):
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = child_lines

        main_str = self._get_name() + '('
        if lines:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def parameters(self, recurse: bool = True):
        r"""Returns an iterator over module parameters.
        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Returns an iterator over all modules in the module, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (string, Module): Tuple of name and module

        """

        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        # memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None:
                    continue
                # memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v
        

class Sequential(Module):
    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None:
        ...
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __len__(self) -> int:
        return len(self._modules)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def append(self, module: Module) -> 'Sequential':
        r"""Appends a given module to the end.

        Args:
            module (Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def forward(self,y,labels,x=None):
        for k,m in enumerate(self._modules):
            idx = labels == k
            self._modules[m](y,x=x,mask=idx)

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

    def sample_A(self):
        tau = 1.0 / np.diag(self.Q)
        Alpha = np.diag(self.alpha)
        for row in range(self.output_dim):
            ell = tau[row] * (self.y[:,row] @ self.x)
            Lam = tau[row] *  ( (self.x.T @ self.x) + Alpha )
            Sigma = la.inv(Lam)
            mu = Sigma @ ell
            self._parameters['A'][row] = mvn.rvs(mu,Sigma)

    def sample_alpha(self):
        if self.hyper_sample is False:
            return 0
        # The marginal prior over A is then a Cauchy distribution: N(a,|0,1/alpha)Gam(alpha|.5,.5) => Cauchy(a)
        a = np.zeros(self.input_dim)
        b = np.zeros(self.input_dim)
        for col in range(self.input_dim):
            a[col] = self.c0 + 1/2*self.output_dim
            b[col] = self.d0 + 1/2*( ((self.A[:,col]) ** 2.0)).sum(0)
        self._parameters['alpha'] = np.atleast_1d(gamma.rvs(a,scale=1.0/b))

    def sample_Q(self):
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
        
    def forward(self,y,x=None,mask=None):
        if x is None:
            x = np.ones((y.shape[0],self.input_dim))
        if mask is None:
            mask = np.ones(y.shape[0]).astype(bool)
        
        self.y = y[mask]
        self.x = x[mask]
        self.N = self.y.shape[0]
        self.sample_A()
        self.sample_Q()
        self.sample_alpha()


class Mixture(Module):
    r'''
        Finite Bayesian mixture.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,components=3):
        super().__init__()
        self.components = components

        self._parameters["z"] = np.ones((1,components)).astype(int)
        self._parameters["pi"] = np.ones(components)/components
        self.alpha0 = np.ones(components) 

    def sample_z(self,logl):
        rho = logl + np.log(self.pi)
        rho -= logsumexp(rho,-1).reshape(-1,1)
        rho = np.exp(rho)
        rho /= rho.sum(-1).reshape(-1,1)
        for n in range(self.N):
            self._parameters['z'][n] = np.random.multinomial(1,rho[n]).argmax()

    def sample_pi(self):
        alpha = np.zeros(self.components)
        for k in range(self.components):
            alpha[k] = (self.z==k).sum() + self.alpha0[k]
        self._parameters['pi'] = dirichlet.rvs(alpha)

    def _check_data(self,logl):
        self.N, components = logl.shape
        if components != self.components:
            raise ValueError("input must have same dimensonality as z (N x C)")
        if self.z.shape[0] != logl.shape[0]:
            self._parameters['z'] = np.random.randint(0,self.components,(self.N))

    def forward(self,logl):
        self._check_data(logl)
        self.sample_z(logl)
        self.sample_pi()

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
        self.theta = Sequential(*[NormalWishart(output_dim=self.output_dim,hyper_sample=self.hyper_sample,full_covariance=self.full_covariance) for i in range(self.components)])
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

    def loglikelihood(self,y):
        N = y.shape[0]
        loglike = np.zeros((N,self.components))
        for k in range(self.components):
            loglike[:,k] = mvn.logpdf(y,self.theta[k].A.ravel(),self.theta[k].Q)
        return loglike

    def _check_input(self,y):
        if y.ndim != 2:
            raise ValueError("input must be 2d")
        N, self.output_dim = y.shape

    def forward(self,y):
        self._check_input(y)
        self.mix(self.loglikelihood(y))
        self.theta(y,self.mix.z)


class HMM(Module):
    r'''
        Bayesian hidden Markov model.

        Gibbs sampling. 

        Author: Julian Neri, December 2022
    '''
    def __init__(self,states=1,expected_duration=1,parameter_sampling=True):
        super().__init__()
        self._dimz = states
        self.expected_duration = expected_duration
        self.parameter_sampling = parameter_sampling

        self._parameters["z"] = np.random.randint(0,self.states,1)
        self.initialize()

    @property
    def states(self):
        return self._dimz
    @states.setter
    def states(self,value):
        value = np.maximum(1,value)
        if value != self._dimz:
            self._dimz = value
            self.initialize()

    def initialize(self):
        A_kk = self.expected_duration / (self.expected_duration+1)
        A_jk = 1.0
        if self.states > 1:
            A_jk = (1-A_kk) / (self.states-1)
        Gamma = np.ones((self.states,self.states)) * A_jk
        np.fill_diagonal(Gamma,A_kk)
        Gamma /= Gamma.sum(-1).reshape(-1,1)
        
        pi = np.ones(self.states) 

        self.Gamma0 = Gamma.copy()
        self.pi0 = pi.copy()

        self._parameters["Gamma"] = Gamma.copy()
        self._parameters["pi"] = pi.copy()

    def _predict_hmm(self,alpha,transpose=False):
        if transpose:
            return np.log(np.exp(alpha) @ self.Gamma.T)
        else:
            return np.log(np.exp(alpha) @ self.Gamma)

    def _forward_hmm(self,logl):
        alpha = np.zeros((self.T,self.states))
        c = np.zeros((self.T))    
        prediction = np.log(self._parameters['pi']).reshape(1,-1)
        for t in range(self.T):
            alpha[t] = logl[t] + prediction
            c[t] = logsumexp(alpha[t])
            alpha[t] -= c[t]
            prediction = self._predict_hmm(alpha[t])
        return np.exp(alpha)

    def _backward_hmm(self,alpha):
        beta = alpha[-1] / alpha[-1].sum()
        self._parameters['z'][-1] = np.random.multinomial(1,beta).argmax()
        for t in range(self.T-2,-1,-1):
            beta = self.Gamma[:,self.z[t+1]] * alpha[t]
            beta /= beta.sum()
            self._parameters['z'][t] = np.random.multinomial(1,beta).argmax()
        
    def sample_z(self,logl):
        alpha = self._forward_hmm(logl)
        self._backward_hmm(alpha)

    def sample_Gamma(self):
        alpha = np.zeros(self.states)
        for k in range(self.states):
            n1 = (self.z[:-1] == k)
            for j in range(self.states):
                n2 = (self.z[1:] == j)
                alpha[j] = self.Gamma0[k,j] + np.sum(n1 & n2)
            self._parameters['Gamma'][k] = dirichlet.rvs(alpha)

    def sample_pi(self):
        alpha = np.zeros(self.states)
        for k in range(self.states):
            alpha[k] = self.pi0[k] + (self.z[0] == k).sum()
        self._parameters['pi'] = dirichlet.rvs(alpha).ravel()

    def _add_data(self,logl):
        self.T, states = logl.shape
        if states != self.states:
            raise ValueError("input logl must have same dimensonality as z (T x S)")
        if self.z.shape[0] != self.T:
            self._parameters['z'] = np.random.randint(0,self.states,self.T)

    def forward(self,logl):
        self._add_data(logl)
        self.sample_z(logl)
        if self.parameter_sampling:
            self.sample_Gamma()
            self.sample_pi()

class GHMM(Module):
    r'''
        Finite Bayesian mixture of Gaussians.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,states=3,hyper_sample=True,full_covariance=True):
        super(GHMM,self).__init__()
        self._dimy = output_dim
        self._dimz = states
        self.hyper_sample = hyper_sample
        self.full_covariance = full_covariance

        self.initialize()

    def initialize(self):
        self.theta = Sequential(*[NormalWishart(output_dim=self.output_dim,hyper_sample=self.hyper_sample,full_covariance=self.full_covariance) for i in range(self.states)])
        self.hmm = HMM(states=self.states)

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
    def states(self):
        return self._dimz
    @states.setter
    def states(self,value):
        value = np.maximum(1,value)
        if value != self._dimz:
            self._dimz = value
            self.initialize()

    def loglikelihood(self,y):
        N = y.shape[0]
        loglike = np.zeros((N,self.states))
        for k in range(self.states):
            loglike[:,k] = mvn.logpdf(y,self.theta[k].A.ravel(),self.theta[k].Q)
        return loglike

    def _check_input(self,y):
        if y.ndim != 2:
            raise ValueError("input must be 2d")
        N, self.output_dim = y.shape

    def forward(self,y):
        self._check_input(y)
        self.hmm(self.loglikelihood(y))
        self.theta(y,self.hmm.z)

class StateSpace(Module):
    def __init__(self,output_dim=1,state_dim=2,hyper_sample=True,full_covariance=True):
        super(StateSpace,self).__init__()
        self._dimy = output_dim
        self._dimx = state_dim
        self.hyper_sample = hyper_sample
        self.full_cov = full_covariance
        self.initialize()
    
    def initialize(self):
        self.sys = NormalWishart(output_dim=self.state_dim, input_dim=self.state_dim,hyper_sample=self.hyper_sample,full_covariance=self.full_cov)
        self.obs = NormalWishart(output_dim=self.output_dim, input_dim=self.state_dim,hyper_sample=self.hyper_sample,full_covariance=self.full_cov)
        self.pri = NormalWishart(output_dim=self.state_dim, input_dim=1,hyper_sample=self.hyper_sample,full_covariance=self.full_cov)

        self.I = np.eye(self.state_dim)

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
    def state_dim(self):
        return self._dimx
    @state_dim.setter
    def state_dim(self,value):
        value = np.maximum(1,value)
        if value != self._dimx:
            self._dimx = value
            self.initialize()

    @property
    def A(self):
        return self.sys.A
    @property
    def Q(self):
        return self.sys.Q
    @property
    def C(self):
        return self.obs.A
    @property
    def R(self):
        return self.obs.Q
    @property
    def m0(self):
        return self.pri.A.ravel()
    @property
    def P0(self):
        return self.pri.Q

    def _check_input(self,y,x):
        if y.shape[0] != x.shape[0]:
            raise ValueError("dim 1 of y and x must match")
        if y.ndim != 2:
            raise ValueError("y must be 2d")
        if x.ndim != 2:
            raise ValueError("x must be 2d")

        self.output_dim = y.shape[-1]
        self.state_dim = x.shape[-1]

    def forward(self,y,x,mask=None):
        self._check_input(y,x)
        if mask is None:
            mask = np.ones(y.shape[0]).astype(bool)
        self.obs(y=y,x=x,mask=mask)
        self.sys(y=x[1:],x=x[:-1],mask=mask[1:])
        self.pri(y=x[[0]],mask=mask[[0]])


class LDS(Module):
    r'''
        Bayesian linear dynamical system.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,state_dim=2,states=1,parameter_sampling=True,hyper_sample=True,full_covariance=True):
        super(LDS,self).__init__()
        self._dimy = output_dim
        self._dimx = state_dim
        self._dimz = states
        self.parameter_sampling = parameter_sampling
        self.hyper_sample = hyper_sample
        self.full_cov = full_covariance

        self._parameters["x"] = np.zeros((1,state_dim))
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

    @property
    def state_dim(self):
        return self._dimx
    @state_dim.setter
    def state_dim(self,value):
        value = np.maximum(1,value)
        if value != self._dimx:
            self._dimx = value
            self.initialize()

    @property
    def states(self):
        return self._dimz
    @states.setter
    def states(self,value):
        value = np.maximum(1,value)
        if value != self._dimz:
            self._dimz = value
            self.initialize()

    def A(self,k):
        return self.theta[k].A
    def Q(self,k):
        return self.theta[k].Q
    def C(self,k):
        return self.theta[k].C
    def R(self,k):
        return self.theta[k].R
    def m0(self,k):
        return self.theta[k].m0
    def P0(self,k):
        return self.theta[k].P0

    def initialize(self):
        self.theta = Sequential(*[StateSpace(output_dim=self.output_dim,state_dim=self.state_dim,hyper_sample=self.hyper_sample,full_covariance=self.full_cov) for i in range(self.states)])
        self.I = np.eye(self.state_dim)
        
    def transition(self,x,z:int=0):
        return self.A(z) @ x
        
    def emission(self,x,z:int=0):
        return self.C(z) @ x
    
    def predict(self,mu,V,z:int=0):
        m = self.transition(mu,z=z)
        P = self.A(z) @ V @ self.A(z).T+ self.Q(z)
        return m, (P)

    def update(self,y,m,P,z:int=0):
        if (y.ndim == 1):
            return self.update_single(y,m,P,z=z)
        else:
            return self.update_multiple(y,m,P,z=z)

    def update_single(self,y,m,P,z:int=0):
        y_hat = self.emission(m,z=z)
        Sigma_hat = self.C(z) @ P @ self.C(z).T + self.R(z)
        K = la.solve(Sigma_hat, self.C(z) @ P).T
        mu = m + K @ (y - y_hat)
        V = (self.I - K @ self.C(z)) @ P
        return mu, (V)

    def update_multiple(self,y,m,P,z:int=0):
        CR = self.C(z).T @ la.inv(self.R(z))
        CRC = CR @ self.C(z)

        Lam = la.inv(P)
        ell = Lam @ m
        ell += CR @ y.sum(0)
        Lam += CRC * y.shape[0]
        V = la.inv(Lam)
        mu = V @ ell
        return mu, (V)

    def _forward(self,y,z):
        mu = np.zeros((self.T,self.state_dim))
        V = np.zeros((self.T,self.state_dim,self.state_dim))
        m = self.m0(z[0]).copy()
        P = self.P0(z[0]).copy()
        for n in range(self.T):
            ''' update'''
            mu[n], V[n] = self.update(y[n],m,P,z=z[n])
            ''' predict '''
            if n < self.T-1:
                m,P = self.predict(mu[n], V[n], z=z[n+1])
        return mu, V

    def _backward(self,mu,V,z):
        self._parameters['x'][-1] = mvn.rvs(mu[-1],V[-1])
        for t in range(self.T-2,-1,-1):
            state = z[t+1]
            m = self.A(state) @ mu[t]
            P = self.A(state) @ V[t] @ self.A(state).T + self.Q(state)
            K_star = V[t] @ self.A(state).T @ la.inv(P)

            mu[t] = mu[t] + K_star @ (self.x[t+1] - m)
            V[t] = (self.I - K_star @ self.A(state)) @ V[t]
            self._parameters['x'][t] = mvn.rvs(mu[t],V[t])

    def sample_x(self,y,z):
        mu, V = self._forward(y,z)
        self._backward(mu,V,z)
        
    def _check_data(self,y,z):
        self.T = y.shape[0]
        self.output_dim = y.shape[1]
        if self.x.shape[0] != self.T:
            self._parameters['x'] = np.zeros((self.T,self.state_dim))

        if z is None:
            # Same state for all time.
            z = np.zeros(self.T).astype(int)
        else:
            if z.shape[0] != self.T:
                raise ValueError("1st dim of z and y must be equal.")
        return z

    def forward(self,y,z=None):
        # z is the state of the system at time t. 
        z = self._check_data(y=y,z=z)
        self.sample_x(y,z=z)
        if self.parameter_sampling == True:
            self.theta(y=y,x=self.x,labels=z)
        
class SLDS(Module):
    r'''
        Bayesian switching linear dynamical system.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,state_dim=2,states=1,parameter_sampling=True,hyper_sample=True,full_covariance=True,expected_duration=10):
        super(SLDS,self).__init__()
        self._dimy = output_dim
        self._dimx = state_dim
        self._dimz = states
        self.parameter_sampling = parameter_sampling
        self.hyper_sample = hyper_sample
        self.full_cov = full_covariance
        self.expected_duration = expected_duration

        self.initialize()

    def initialize(self):
        self.hmm = HMM(states=self.states,expected_duration=self.expected_duration,parameter_sampling=self.parameter_sampling)
        self.lds = LDS(output_dim=self.output_dim,state_dim=self.state_dim,states=self.states,full_covariance=self.full_cov,parameter_sampling=self.parameter_sampling,hyper_sample=self.hyper_sample)

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
    def state_dim(self):
        return self._dimx
    @state_dim.setter
    def state_dim(self,value):
        value = np.maximum(1,value)
        if value != self._dimx:
            self._dimx = value
            self.initialize()

    @property
    def states(self):
        return self._dimz
    @states.setter
    def states(self,value):
        value = np.maximum(1,value)
        if value != self._dimz:
            self._dimz = value
            self.initialize()

    def loglikelihood(self,y):
        logl = np.zeros((self.T, self.states))
        for s in range(self.states):
            mu =  self.lds.x @ self.lds.C(s).T
            Sigma = self.lds.R(s)
            logl[:,s] = mvn_logpdf(y,mu,Sigma)

            m = self.lds.m0(s)[None,:]
            P = self.lds.P0(s)
            logl[0,s] += mvn_logpdf(self.lds.x[[0]],m,P)
 
            m = self.lds.x[:-1] @ self.lds.A(s).T 
            P = self.lds.Q(s)
            logl[1:,s] += mvn_logpdf(self.lds.x[1:],m,P)

        return logl

    def _check_data(self,y):
        self.T = y.shape[0]
        self.output_dim = y.shape[1]
        if self.hmm.z.shape[0] != self.T:
            self.hmm._parameters['z'] = np.random.randint(0,self.states,self.T)

    def forward(self,y):
        self._check_data(y=y)
        self.lds(y=y,z=self.hmm.z)
        self.hmm(logl=self.loglikelihood(y))