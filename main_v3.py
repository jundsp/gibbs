#%%
from gibbs import get_mean, get_median, plot_cov_ellipse, gmm_generate, hmm_generate
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import gamma, wishart, dirichlet
import scipy.linalg as la
from scipy.special import logsumexp
from typing import OrderedDict, Optional, Iterable, overload, Set
from itertools import islice
import operator
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.modules.container


class Gibbs(object):
    r'''
    Gibbs sampler base class.

    Author: Julian Neri, May 2022
    '''
    def __init__(self):
        self._samples = OrderedDict()
        self._estimates = OrderedDict()

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

    def step(self,params):
        for name,value in params:
            if name not in self._samples.keys():
                self._samples[name] = []
            self._samples[name].append(value.copy())

    def fit(self,sampler_fn, params_fn,samples=100):
        for iter in tqdm(range(samples)):
            sampler_fn()
            self.step(params_fn())


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
            self.__dict__[__name] = __value

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

    def forward(self,data: 'Data',labels):
        for k,m in enumerate(self._modules):
            idx = labels == k
            _data = Data(y=data.y[idx],t=data.loc[idx])
            _data.T = data.T+0
            self._modules[m](_data)



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
        
    def forward(self,data: 'Data'):
        self.y = data.y
        if data.x is None:
            x = np.ones((self.y.shape[0],self.input_dim))
            
        self.x = x
        self.N = y.shape[0]
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

    def sample_z(self,loglike):
        rho = loglike + np.log(self._parameters['pi'])
        rho -= logsumexp(rho,-1).reshape(-1,1)
        rho = np.exp(rho)
        rho /= rho.sum(-1).reshape(-1,1)
        for n in range(self.N):
            self._parameters['z'][n] = np.random.multinomial(1,rho[n]).argmax()

    def sample_pi(self):
        alpha = np.zeros(self.components)
        for k in range(self.components):
            alpha[k] = (self._parameters['z']==k).sum() + self.alpha0[k]
        self._parameters['pi'] = dirichlet.rvs(alpha)

    def _add_data(self,loglike):
        self.N, components = loglike.shape
        if components != self.components:
            raise ValueError("input must have same dimensonality as z (N x C)")
        if self.z.shape[0] != self.N:
            self._parameters['z'] = np.random.randint(0,self.components,self.N)

    def forward(self,loglike):
        self._add_data(loglike)
        self.sample_z(loglike)
        self.sample_pi()

class GMM(Module):
    r'''
        Finite Bayesian mixture of Gaussians.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,components=3,hyper_sample=True,full_covariance=True):
        super().__init__()
        self.output_dim = output_dim
        self.components = components

        self.theta = Sequential(*[NormalWishart(output_dim=output_dim,hyper_sample=hyper_sample,full_covariance=full_covariance) for i in range(components)])
        self.mix = Mixture(components=components)

    def loglikelihood(self,data):
        N = len(data)
        loglike = np.zeros((N,self.components))
        for k in range(self.components):
            loglike[:,k] = mvn.logpdf(data.y,self.theta[k].A.ravel(),self.theta[k].Q)
        return loglike

    def _check_input(self,data: 'Data'):
        if data.y.shape[-1] != self.output_dim:
            raise ValueError("input dimension does not match model")

    def forward(self,data: 'Data'):
        self._check_input(data)
        self.mix(self.loglikelihood(data))
        self.theta(data,self.mix.z)


class HMM(Module):
    r'''
        Bayesian hidden Markov model.

        Gibbs sampling. 

        Author: Julian Neri, December 2022
    '''
    def __init__(self,states=1,expected_duration=1):
        super().__init__()
        self._dimz = states
        self.expected_duration = expected_duration

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
        
        pi = np.ones(self.states) / self.states

        self.Gamma0 = Gamma.copy()
        self.pi0 = pi.copy()

        self._parameters["Gamma"] = Gamma.copy()
        self._parameters["pi"] = pi.copy()

    def _predict_hmm(self,alpha,transpose=False):
        if transpose:
            return np.log(np.exp(alpha) @ self.Gamma.T)
        else:
            return np.log(np.exp(alpha) @ self.Gamma)

    def _forward_hmm(self,loc,loglikelihood):
        alpha = np.zeros((self.T,self.states))
        c = np.zeros((self.T))    
        prediction = np.log(self._parameters['pi']).reshape(1,-1)
        for t in range(self.T):
            alpha[t] = loglikelihood[loc==t].sum(0) + prediction
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
        
    def sample_z(self,loc,loglikelihood):
        alpha = self._forward_hmm(loc,loglikelihood)
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

    def _check_data(self,T,loglikelihood):
        self.T = T
        if loglikelihood.shape[-1] != self.states:
            raise ValueError("input must have same dimensonality as z (T x S)")
        if self.z.shape[0] != self.T:
            self._parameters['z'] = np.random.randint(0,self.states,self.T)

    def forward(self,data: 'Data',loglikelihood):
        self._check_data(data.T,loglikelihood)
        self.sample_z(data.loc,loglikelihood)
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
        self.output_dim = output_dim
        self.states = states

        self.theta = Sequential(*[NormalWishart(output_dim=output_dim,hyper_sample=hyper_sample,full_covariance=full_covariance) for i in range(states)])
        self.hmm = HMM(states=states)

    def loglikelihood(self,data: 'Data'):
        loglike = np.zeros((len(data),self.states))
        for k in range(self.states):
            loglike[:,k] = mvn.logpdf(data.y,self.theta[k].A.ravel(),self.theta[k].Q)
        return loglike

    def logmarginal(self,data: 'Data'):
        loglike = self.loglikelihood(data)
        z_flat = self.hmm.z[data.loc]
        marg = loglike[np.arange(len(data)),z_flat]
        return marg
        
    def _check_input(self,data: 'Data'):
        if data.y.shape[-1] != self.output_dim:
            raise ValueError("input dimension does not match model")

    def forward(self,data: 'Data'):
        self._check_input(data)
        self.hmm(data,self.loglikelihood(data))
        labels = np.zeros(len(data)).astype(int)
        for i in range(len(data)):
            labels[i] = self.hmm.z[data.loc[i]]
        self.theta(data,labels)


class MGHMM(Module):
    r'''
        Finite Bayesian mixture of HMMs.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,components=3,states=1,hyper_sample=True,full_covariance=True):
        super(MGHMM,self).__init__()
        self.output_dim = output_dim
        self.components = components
        self.states = states

        self.ghmm = Sequential(*[GHMM(output_dim=output_dim,states=states,hyper_sample=hyper_sample,full_covariance=full_covariance) for i in range(components)])
        self.mix = Mixture(components=components)

    def loglikelihood(self,data: 'Data'):
        N = len(data)
        loglike = np.zeros((N,self.components))
        for k in range(self.components):
            loglike[:,k] = self.ghmm[k].logmarginal(data)
        return loglike

    def _check_input(self,data: 'Data'):
        if data.y.shape[-1] != self.output_dim:
            raise ValueError("input dimension does not match model")
        initial_z = np.random.randint(0,self.components,len(data))
        self.ghmm(data,initial_z)

    def forward(self,data: 'Data'):
        self._check_input(data)
        self.mix(self.loglikelihood(data))
        self.ghmm(data,self.mix.z)
        
        

#* output, input, time
class Data(object):
    def __init__(self,y,x=None,t=None,delta=None) -> None:
        self.y = y
        self.x = x
        self.loc = t
        
    @property
    def y(self):
        return self._y

    @y.setter
    def y(self,val):
        if val.ndim != 2:
            raise ValueError("value of y for 'Data' must be 2d")
        self._y = val

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self,val):
        self._x = val

    @property
    def loc(self):
        return self._t

    @loc.setter
    def loc(self,val):
        if val is None:
            val = np.arange(self.y.shape[0])
        val = np.atleast_1d(val)
        if val.ndim != 1:
            raise ValueError("value of loc for 'Data' must be 1d")
        if len(val) != len(self):
            raise ValueError("len of loc must match len of y for 'Data'")
        if len(val) == 0:
            self.T = 0
        else:
            self.T = int(val.max() + 1)
        self._t = val

    def __len__(self):
        return self.y.shape[0]


#%%
N = 500
np.random.seed(123)
y,z_true = hmm_generate(n=N,n_components=2,expected_duration=20)
t = np.arange(N)
delta = np.ones(N).astype(bool)
# delta[:200] = False
y = y[delta]
t = t[delta]

y = np.concatenate([y,-2*y],0)
t = np.concatenate([t,t],0)
sot = t.argsort()
y = y[sot]
t = t[sot]
data = Data(y,t=t)

plt.plot(data.loc,data.y,'.')
#%%
np.random.seed(123)
model = MGHMM(output_dim=2,states=4,components=2,hyper_sample=True)
sampler = Gibbs()

#%%
samples = 100
for i in tqdm(range(samples)):
    model(data)
    sampler.step(model.named_parameters())

#%%
sampler.get_estimates(burn_rate=.8)
chain = sampler.get_chain(flatten=True)

fig,ax = plt.subplots(len(chain),figsize=(5,len(chain)*1.5))
ax = np.atleast_1d(ax)
for j,s in enumerate(chain):
    ax[j].plot(chain[s],'k',alpha=.1,linewidth=1)
    ax[j].set_title(s)
ax[-1].set_xlabel('step')
plt.tight_layout()

#%%
colors = np.array(['r','g','b','k','m','orange','yellow','brown']*30)
plt.figure(figsize=(5,4))
# z_hat = sampler._estimates['hmm.z']
# labels = np.zeros(len(data)).astype(int)
# for i in range(len(data)):
#     labels[i] = z_hat[data.loc[i]]
z_hat = sampler._estimates['mix.z']
labels = z_hat + 0

    
kwds_scatter = dict(s=15,alpha=.5,edgecolor='none')
plt.scatter(data.y[:,0],data.y[:,1],c=colors[labels],**kwds_scatter)

for k in np.unique(z_hat):
    for l in np.unique(sampler._estimates[r"ghmm.{}.hmm.z".format(k)]):
        mu_hat = sampler._estimates[r"ghmm.{}.theta.{}.A".format(k,l)]
        Sigma_hat = sampler._estimates[r"ghmm.{}.theta.{}.Q".format(k,l)]
        plot_cov_ellipse(pos=mu_hat,cov=Sigma_hat,nstd=1,fill=None,color=colors[k],linewidth=2)
plt.tight_layout()

fig,ax = plt.subplots(2)
ax[0].imshow(np.atleast_2d(z_true),aspect='auto')
ax[1].imshow(np.atleast_2d(z_hat),aspect='auto')


# %%
