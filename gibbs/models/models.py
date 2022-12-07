import numpy as np
from scipy.stats import multivariate_t, norm, wishart, multinomial, dirichlet, gamma
from scipy.stats import multivariate_normal as mvn
from scipy.special import logsumexp
import scipy.linalg as la
import matplotlib.pyplot as plt

from ..core import GibbsDirichletProcess, DirichletProcess, Gibbs
from ..utils import plot_cov_ellipse

class GMM(Gibbs):
    r'''
        Finite Bayesian mixture of Gaussians.

        Gibbs sampling. 

        Author: Julian Neri, 2022

        Examples
        --------
        Import package
        >>> from gibbs import GMM, gmm_generate

        Generate data
        >>> x = gmm_generate(200)[0]
        Create model
        >>> model = GMM(output_dim=2)
        Fit the model to the data using the Gibbs sampler.
        >>> model.fit(x,samples=20)
        >>> model.plot()
        >>> model.plot_samples()
    '''
    def __init__(self,output_dim=1,components=3):
        super().__init__()
        self.output_dim = output_dim
        self.components = components
        
        self.register_parameter("z",None)
        self.register_parameter("mu",3*np.random.normal(0,1,(components,output_dim)))
        self.register_parameter("Sigma",np.stack([np.eye(output_dim)]*components,0))
        self.register_parameter("pi",np.ones(components)/components)
        
        self.lambda0 = .01
        self.alpha0 = np.ones(components) / components
        self.nu0 = self.output_dim + 1.0
        self.iW0 = np.eye(self.output_dim)
        
    def fit(self,x,samples=100):
        self.x = x
        self.N = x.shape[0]
        if self._parameters['z'] is None:
            self._parameters['z'] = np.random.randint(0,self.components,self.N)
        super().fit(samples)

    def loglikelihood(self,x):
        N = x.shape[0]
        loglike = np.zeros((N,self.components))
        for k in range(self.components):
            loglike[:,k] = mvn.logpdf(x,self._parameters['mu'][k],self._parameters['Sigma'][k])
        return loglike

    def _sample_z(self):
        rho = self.loglikelihood(self.x) + np.log(self._parameters['pi'])
        rho -= logsumexp(rho,-1).reshape(-1,1)
        rho = np.exp(rho)
        rho /= rho.sum(-1).reshape(-1,1)
        for n in range(self.N):
            self._parameters['z'][n] = np.random.multinomial(1,rho[n]).argmax()

    def _sample_pi(self):
        alpha = np.zeros(self.components)
        for k in range(self.components):
            alpha[k] = (self._parameters['z']==k).sum() + self.alpha0[k]
        self._parameters['pi'] = dirichlet.rvs(alpha)

    def _sample_mu(self):
        for k in range(self.components):
            idx = (self._parameters['z']==k)
            Nk = idx.sum()
            iSigma = la.inv(self._parameters['Sigma'][k])
            ell = iSigma @ self.x[idx].sum(0) 
            lam = iSigma * Nk  + self.lambda0*np.eye(ell.shape[0])
            Sigma = la.inv(lam)
            mean = Sigma @ ell
            self._parameters['mu'][k] = mvn.rvs(mean, Sigma)

    def _sample_Sigma(self):
        for k in range(self.components):
            idx = (self._parameters['z']==k)
            
            Nk = idx.sum()
            nu = self.nu0 + Nk

            x_eps = self.x[idx] - self._parameters['mu'][k][None,:]
            iW = self.iW0 + x_eps.T @ x_eps
            W = la.inv(iW)

            Lambda = wishart.rvs(df=nu,scale=W)      
            self._parameters['Sigma'][k] = la.inv(Lambda)    

    def plot(self,figsize=(4,3),**kwds_scatter):
        z_hat = self._estimates['z'].astype(int)
        colors = np.array(['r','g','b','m','y','k','orange']*3)
        plt.figure(figsize=figsize)
        if self.output_dim == 2:
            plt.scatter(self.x[:,0],self.x[:,1],c=colors[z_hat],**kwds_scatter)
            for k in np.unique(z_hat):
                plot_cov_ellipse(self._estimates['mu'][k],self._estimates['Sigma'][k],facecolor='none',edgecolor=colors[k])
        plt.xlabel('$y_1$')
        plt.ylabel('$y_2$')
        plt.tight_layout()

class HMM(Gibbs):
    r'''
        Bayesian hidden Markov model, Gaussian emmission.

        Gibbs sampling. 

        Author: Julian Neri, 2022

        Examples
        --------
        Import package
        >>> from gibbs import HMM, hmm_generate

        Generate data
        >>> x = hmm_generate(200)[0]
        Create model
        >>> model = HMM(output_dim=2)
        Fit the model to the data using the Gibbs sampler.
        >>> model.fit(x,samples=20)
        >>> model.plot()
        >>> model.plot_samples()
    '''
    def __init__(self,output_dim=1,switch_dim=1,expected_duration=5,parameter_sampling=True):
        super().__init__()
        self._dimy = output_dim
        self._dimz = switch_dim
        self.expected_duration = expected_duration
        self.parameter_sampling = parameter_sampling

        self.register_parameter("z",None)
        self.initialize()

    @property
    def output_dim(self):
        return self._dimy
    @output_dim.setter
    def output_dim(self,value):
        value = np.maximum(1,value)
        if value != self._dimy:
            self._dimy = value
            self.initialize_output_model()

    @property
    def switch_dim(self):
        return self._dimz
    @switch_dim.setter
    def switch_dim(self,value):
        value = np.maximum(1,value)
        if value != self._dimz:
            self._dimz = value
            self.initialize_switch_model()

    @property
    def state_dim(self):
        return self._dimx
    @state_dim.setter
    def state_dim(self,value):
        value = np.maximum(1,value)
        if value != self._dimx:
            self._dimx = value
            self.initialize_state_model()
            self.initialize_prior_model()
            
    @property
    def output(self):
        y_hat = np.zeros((self.T,self.output_dim))
        for t in range(self.T):
            y_hat[t] = self._emission(t)
        return y_hat

    def initialize(self):
        self.initialize_switch_model()
        self.initialize_output_model()

    def initialize_switch_model(self):
        A_kk = self.expected_duration / (self.expected_duration+1)
        A_jk = 1.0
        if self.switch_dim > 1:
            A_jk = (1-A_kk) / (self.switch_dim-1)
        Gamma = np.ones((self.switch_dim,self.switch_dim)) * A_jk
        np.fill_diagonal(Gamma,A_kk)
        Gamma /= Gamma.sum(-1).reshape(-1,1)
        
        pi = np.ones(self.switch_dim) / self.switch_dim

        self.prior_Gamma = Gamma.copy()
        self.prior_pi = pi.copy()

        self.register_parameter("Gamma",Gamma.copy())
        self.register_parameter("pi",pi.copy())

    def initialize_output_model(self):
        mu0 = np.zeros(self.output_dim)
        Sigma0 = np.eye(self.output_dim)

        self.nu0 = self.output_dim+1
        self.iW0 = np.eye(self.output_dim)
        self.lambda0 = 1e-2

        mu = mvn.rvs(mu0,Sigma0,self.switch_dim)
        mu = np.atleast_2d(mu)
        self.register_parameter("mu",mu.copy())
        self.register_parameter("Sigma",np.stack([Sigma0]*self.switch_dim,0))

    def _sample_mu(self):
        if self.parameter_sampling is False:
            return 0

        for k in range(self.switch_dim):
            m = 0
            Nk = 0
            for t in range(self.T):
                idx = (self.z[t]==k) & self.delta[t]
                Nk += idx.sum()
                m += self.y[t,idx].sum(0)

            iSigma = la.inv(self._parameters['Sigma'][k])
            ell = iSigma @ m
            lam = iSigma * Nk  + self.lambda0*np.eye(ell.shape[0])
            Sigma = la.inv(lam)
            mean = Sigma @ ell
            self._parameters['mu'][k] = mvn.rvs(mean, Sigma)

    def _sample_Sigma(self):
        if self.parameter_sampling is False:
            return 0

        for k in range(self.switch_dim):
            nu = self.nu0 + 0
            iW = self.iW0 + 0
            for t in range(self.T):
                idx = (self.z[t]==k) & self.delta[t]
                nu += idx.sum()
                x_eps = self.y[t,idx] - self._parameters['mu'][k][None,:]
                iW += x_eps.T @ x_eps

            W = la.inv(iW)
            Lambda = wishart.rvs(df=nu,scale=W)      
            self._parameters['Sigma'][k] = la.inv(Lambda)  

    def _sample_Gamma(self):
        if self.parameter_sampling is False:
            return 0
        alpha = np.zeros(self.switch_dim)
        for k in range(self.switch_dim):
            n1 = (self._parameters['z'][:-1] == k)
            for j in range(self.switch_dim):
                n2 = (self._parameters['z'][1:] == j)
                alpha[j] = self.prior_Gamma[k,j] + np.sum(n1 & n2)
            self._parameters['Gamma'][k] = dirichlet.rvs(alpha)

    def _sample_pi(self):
        if self.parameter_sampling is False:
            return 0
        alpha = np.zeros(self.switch_dim)
        for k in range(self.switch_dim):
            alpha[k] = self.prior_pi[k] + (self._parameters['z'][0] == k).sum()
        self._parameters['pi'] = dirichlet.rvs(alpha).ravel()

    def _predict_hmm(self,alpha,transpose=False):
        if transpose:
            return np.log(np.exp(alpha) @ self._parameters['Gamma'].T)
        else:
            return np.log(np.exp(alpha) @ self._parameters['Gamma'])

    def _log_emission_hmm(self,t):
        logpr = np.zeros(self.switch_dim)
        for k in range(self.switch_dim):
            for m in np.nonzero(self.delta[t])[0]:
                logpr[k] += mvn.logpdf(self.y[t,m],self._parameters['mu'][k] ,self._parameters['Sigma'][k])
        return logpr

    def _forward_hmm(self):
        alpha = np.zeros((self.T,self.switch_dim))
        c = np.zeros((self.T))    
        prediction = np.log(self._parameters['pi']).reshape(1,-1)
        for t in range(self.T):
            alpha[t] = self._log_emission_hmm(t) + prediction
            c[t] = logsumexp(alpha[t])
            alpha[t] -= c[t]
            prediction = self._predict_hmm(alpha[t])
        return np.exp(alpha)
        
    def _sample_z(self):
        alpha = self._forward_hmm()
        beta = alpha[-1] / alpha[-1].sum()
        self._parameters['z'][-1] = np.random.multinomial(1,beta).argmax()
        for t in range(self.T-2,-1,-1):
            beta = self._parameters['Gamma'][:,self._parameters['z'][t+1]] * alpha[t]
            beta /= beta.sum()
            self._parameters['z'][t] = np.random.multinomial(1,beta).argmax()

    def add_data(self,y,delta=None):
        if y.ndim == 1:
            y = y.reshape(-1,1,1)
        elif y.ndim == 2:
            y = np.expand_dims(y,1)

        self.y = y.copy()
        self.T, self.N, self.output_dim = y.shape

        if delta is None:
            delta = np.ones((self.T,self.N))
        if delta.ndim == 1:
            delta = delta[:,None] + np.zeros((1,self.N))
        self.delta = delta.astype(bool).copy()

    def init_samples(self):
        if self._parameters["z"] is None:
            self._parameters["z"] = np.random.randint(0,self.switch_dim,(self.T))

    def fit(self,y,delta=None,samples=10):
        self.add_data(y=y,delta=delta)
        self.init_samples()
        super().fit(samples)

    def generate(self,n=100):
        z = np.zeros(n).astype(int)
        y = np.zeros((n,self.output_dim))
        predict = self.pi.copy()
        for i in range(n):
            z[i] = multinomial.rvs(1,predict).argmax()
            y[i] = np.random.multivariate_normal(self.mu[z[i]],self.Sigma[z[i]])
            predict = self.Gamma[z[i]]
        return y, z

    def plot(self,figsize=(4,3),**kwds_scatter):
        z_hat = self._estimates['z'].astype(int)
        colors = np.array(['r','g','b','m','y','k','orange']*3)
        plt.figure(figsize=figsize)
        if self.output_dim == 2:
            for n in range(self.N):
                plt.scatter(self.y[:,n,0],self.y[:,n,1],c=colors[z_hat],**kwds_scatter)
            for k in np.unique(z_hat):
                plot_cov_ellipse(self._estimates['mu'][k],self._estimates['Sigma'][k],facecolor='none',edgecolor=colors[k])
        plt.xlabel('$y_1$')
        plt.ylabel('$y_2$')
        plt.tight_layout()

class DP_GMM(GibbsDirichletProcess):
    r'''
    Infinite Gaussian mixture model.

    Dirichlet process, gibbs sampling. 

    Author: Julian Neri, 2022

    Examples
    --------
    Import package
    >>> from gibbs import DP_GMM

    Create model and generate data from it
    >>> model = DP_GMM(output_dim=2)
    >>> x = model.generate(200)[0]

    Fit the model to the data using the Gibbs sampler.
    >>> model.fit(x,samples=20)
    >>> model.plot()
    >>> model.plot_samples()

    '''
    def __init__(self,output_dim=1,alpha=1,learn=True,outliers=False):
        super().__init__(alpha=alpha,learn=learn)
        self.outliers = outliers
        self.output_dim = output_dim
        self._hyperparameters = []
        self._kinds = []
        self._init_prior()
  
    def _init_prior(self):
        self._hyperparameter_lookup = [[],[]]
        

        m0 = np.zeros(self.output_dim)
        nu0 = (self.output_dim+1)
        k0 = .01
        S0 = np.eye(self.output_dim)*nu0
        self._hyperparameter_lookup[0] = tuple([m0,k0,S0,nu0])

        m0 = np.zeros(self.output_dim)
        nu0 = self.output_dim+1
        k0 = .01
        S0 = np.eye(self.output_dim)*nu0*.1
        self._hyperparameter_lookup[1] = tuple([m0,k0,S0,nu0])

    def _hyperparameter_sample(self):

        m0 = np.zeros(self.output_dim)
        nu0 = self.output_dim+1
        k0 = .01
        sigma = 1/np.random.gamma(1,5)
        S0 = np.eye(self.output_dim)*nu0*sigma
        
        return m0,k0,S0,nu0

    def _posterior(self,idx,m0,k0,S0,nu0):
        N = idx.sum()
        xk = np.atleast_2d(self.x[idx])
        x_bar = xk.mean(0)
        S = xk.T @ xk
        kN = k0 + N
        mN = (k0 * m0 + x_bar * N)/kN
        nuN = nu0 + N
        SN = S0 + S + k0*np.outer(m0,m0) - kN*np.outer(mN,mN)
        return mN,kN,SN,nuN

    def _posterior_predictive(self,x_i,idx,m0,k0,S0,nu0):
        m,k,S,nu = self._posterior(idx,m0,k0,S0,nu0)
        return self._predictive(x_i,m,k,S,nu)

    def _prior_predictive(self,x_i,m0,k0,S0,nu0):
        return self._predictive(x_i,m0,k0,S0,nu0)

    def _predictive_parameters(self,m,k,S,nu):
        _m = m
        _nu = nu - self.output_dim + 1.0
        _S = (k + 1.0)/(k*_nu) * S
        return _m, _S, _nu

    # Multivariate t
    def _predictive(self,x,m,k,S,nu):
        _m, _S, _nu = self._predictive_parameters(m,k,S,nu)
        return multivariate_t.pdf(x,loc=_m,shape=_S,df=_nu)

    def _get_rho_one(self,n,k):
        idx = self._parameters['z'] == k
        idx[n] = False
        Nk = idx.sum() 

        rho = 0.0
        if Nk > 0:
            m0,k0,S0,nu0 = self._hyperparameters[k]
            rho = self._posterior_predictive(self.x[n],idx,m0,k0,S0,nu0) * Nk
        return rho

    def _sample_z_one(self,n):
        denominator = self.N + self._parameters['alpha'] - 1
        K_total = self.K + 1
        if self.outliers: K_total += 1

        rho = np.zeros(K_total)
        for k in range(self.K):
            rho[k] = self._get_rho_one(n,k)

        theta_prior = [[],[]]
        if self.outliers:
            m0,k0,S0,nu0 = self._hyperparameter_lookup[0]
            theta_prior[0] = tuple([m0,k0,S0,nu0])
            rho[-2] = self._prior_predictive(self.x[n],m0,k0,S0,nu0) * self._parameters['alpha'] 

        m0,k0,S0,nu0 = self._hyperparameter_lookup[1]
        theta_prior[1] = tuple([m0,k0,S0,nu0])
        rho[-1] = self._prior_predictive(self.x[n],m0,k0,S0,nu0) * self._parameters['alpha'] 

        rho /= rho.sum()
        _z_now = np.random.multinomial(1,rho).argmax()
        
        if _z_now > (self.K-1):
            kind = 1
            if self.outliers:
                if _z_now == self.K:
                    kind = 0
            _z_now = self.K + 0
            self._hyperparameters.append(theta_prior[kind])
            self._kinds.append(kind)
            self.K += 1
        self._parameters['z'][n] = _z_now
        
    def _sample_z(self):
        tau = np.random.permutation(self.N)
        for n in tau:
            self._sample_z_one(n)
        self._collapse_groups()

    def generate(self,n=100):
        dp = DirichletProcess(alpha=self._parameters['alpha'])
        mu = np.random.normal(0,2,(100,self.output_dim))
        sigmas = np.random.gamma(shape=3,scale=.1,size=100)
        sigmas[0] = 2
        Sigma = np.stack([np.eye(self.output_dim)*s for s in sigmas],0)

        z = np.zeros(n).astype(int)
        x = np.zeros((n,self.output_dim))
        for i in range(n):
            z[i] = dp.sample()
            x[i] = mvn.rvs(mu[z[i]],Sigma[z[i]])
        return x, z

    def fit(self,x,samples=100):
        self.x = x
        self.N = x.shape[0]
        if x.shape[-1] != self.output_dim:
            self.output_dim = x.shape[-1]
            self._init_prior()
        if self._parameters['z'] is None:
            self._parameters['z'] = np.zeros(self.N).astype(int) - 1
            self.K = 0
        super().fit(samples)

    def plot(self,figsize=(5,4),**kwds_scatter):
        z_hat = self._parameters['z'].astype(int)
        K_hat = np.unique(z_hat)
        colors = np.array(['r','g','b','m','y','k','orange']*30)
        plt.figure(figsize=figsize)
        if self.output_dim == 2:
            plt.scatter(self.x[:,0],self.x[:,1],c=colors[z_hat],**kwds_scatter)
            for k in (K_hat):
                if self._kinds[k] == 0:
                    linestyle = 'dashed'
                else:
                    linestyle = 'solid'
                idx = z_hat==k
                m0,k0,S0,nu0 = self._hyperparameters[k]
                muN,kN,SN,nuN = self._posterior(idx,m0,k0,S0,nu0)
                mu_x, S_x, nu_x = self._predictive_parameters(muN,kN,SN,nuN)
                S_x *= nu_x/(nu_x - 2)

                plot_cov_ellipse(mu_x,S_x,facecolor='none',edgecolor=colors[k],linestyle=linestyle)

            plt.xlabel('$y_1$')
            plt.ylabel('$y_2$')
        elif self.output_dim == 1:
            _x = np.linspace(self.x.min(),self.x.max(),128)
            plt.scatter(self.x,self.x*0,c=colors[z_hat])
            for k in (K_hat):
                idx = z_hat==k
                m0,k0,S0,nu0 = self._hyperparameters[k]
                muN,kN,SN,nuN = self._posterior(idx,m0,k0,S0,nu0)
                mu_x, S_x, nu_x = self._predictive_parameters(muN,kN,SN,nuN)
                S_x *= nu_x/(nu_x - 2)
                _px = norm.pdf(_x,mu_x.ravel(),S_x.ravel())
                plt.plot(_x,_px,color=colors[k])
        plt.tight_layout()



class NormalParameters(Gibbs):
    r'''
        Bayesian linear dynamical system parameters

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,input_dim=1,parameter_sampling=True,hyperparameter_sampling=True,system_covariance=True):
        super(NormalParameters,self).__init__()
        self._dimo = output_dim
        self._dimi = input_dim
        self.parameter_sampling = parameter_sampling
        self.hyperparameter_sampling = hyperparameter_sampling
        self.system_cov = system_covariance

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
        A = np.eye(self.output_dim,self.input_dim)
        A += np.random.normal(0,1e-3,A.shape)
        A /= np.abs(A).sum(-1)[:,None]
        Q = np.eye(self.output_dim)
        alpha = np.ones((self.input_dim))*1e-1

        self.register_parameter("A",A.copy())
        self.register_parameter("Q",Q.copy())
        self.register_parameter('alpha',alpha.copy())

        nu = self.output_dim + 1.0
        v = 1.0

        self.a0 = nu / 2.0
        self.b0 = 1.0 / (2*v)

        self.A0 = np.eye(self.output_dim,self.input_dim)*0

    def _sample_A(self):
        if self.parameter_sampling is False:
            return 0

        tau = 1.0 / np.diag(self.Q)
        Alpha = np.diag(self.alpha)
        for row in range(self.output_dim):
            ell = tau[row] * (self.y[:,row] @ self.x + Alpha @ self.A0[row])
            Lam = tau[row] *  ( (self.x.T @ self.x) + Alpha )
            Sigma = la.inv(Lam)
            mu = Sigma @ ell
            self._parameters['A'][row] = mvn.rvs(mu,Sigma)

    def _sample_Q(self):
        if self.parameter_sampling is False:
            return 0

        Nk = self.N + self.input_dim
        x_eps = self.y - self.x @ self.A.T
        quad = np.diag(x_eps.T @ x_eps) + np.diag((self.A-self.A0) @ np.diag(self.alpha) @ (self.A-self.A0).T)

        a_hat = self.a0 + 1/2 * Nk
        b_hat = self.b0 + 1/2 * quad

        tau = np.atleast_1d(gamma.rvs(a_hat,scale=1/b_hat))
        self._parameters['Q'] = np.diag(1/tau)

    def _sample_alpha(self):
        if self.hyperparameter_sampling is False:
            return 0

        # The marginal prior over A is then a Cauchy distribution: N(a,|0,1/alpha)Gam(alpha|.5,.5) => Cauchy(a)
        a0, b0 = .5,.5
        a = np.zeros(self.input_dim)
        b = np.zeros(self.input_dim)
        tau = 1.0 / np.diag(self.Q)
        for col in range(self.input_dim):
            a[col] = a0 + 1/2*self.output_dim
            b[col] = b0 + 1/2*( ((self.A[:,col]-self.A0[:,col]) ** 2.0)).sum(0)
        self._parameters['alpha'] = np.atleast_1d(gamma.rvs(a,scale=1.0/b))

    def add_data(self,y,x):
        if x.ndim != 2:
            raise ValueError('input must be 2d')
        if y.ndim != 2:
            raise ValueError('output must be 2d')
        if x.ndim != y.ndim:
            raise ValueError("# of dims of output must equal input")
        if x.shape[0] != y.shape[0]:
            raise ValueError("First dim of output and input must be equal")

        self.x = x.copy()
        self.y = y.copy()
        self.input_dim = x.shape[-1]
        self.N, self.output_dim = y.shape

    def fit(self,y,x,samples=10):
        self.add_data(y=y,x=x)
        super().fit(samples)



class NormalWishParams(Gibbs):
    r'''
        Bayesian linear dynamical system parameters

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,input_dim=1,parameter_sampling=True,hyperparameter_sampling=True,system_covariance=True):
        super(NormalWishParams,self).__init__()
        self._dimo = output_dim
        self._dimi = input_dim
        self.parameter_sampling = parameter_sampling
        self.hyperparameter_sampling = hyperparameter_sampling
        self.system_cov = system_covariance

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
        A = np.eye(self.output_dim,self.input_dim)
        A += np.random.normal(0,1e-3,A.shape)
        A /= np.abs(A).sum(-1)[:,None]
        Q = np.eye(self.output_dim)
        alpha = np.ones((self.input_dim))*1e-1

        self.register_parameter("A",A.copy())
        self.register_parameter("Q",Q.copy())
        self.register_parameter('alpha',alpha.copy())

        self.nu0 = self.output_dim+1
        self.iW0 = np.eye(self.output_dim)

    def _sample_A(self):
        if self.parameter_sampling is False:
            return 0

        Qi = la.inv(self.Q)
        Alpha = np.diag(self.alpha)
        for row in range(self.output_dim):
            ell = Qi[row,row] * (self.y[:,row].T @ self.x)
            Lam = Qi[row,row] * (self.x.T @ self.x)
            Sigma = la.inv(Lam + Alpha)
            mu = Sigma @ ell
            self._parameters['A'][row] = mvn.rvs(mu,Sigma)

    def _sample_Q(self):
        if self.parameter_sampling is False:
            return 0

        nu = self.nu0 + self.N

        y_hat = (self.x @ self.A.T)
        x_eps = (self.y - y_hat)
        quad = x_eps.T @ x_eps
        iW = self.iW0 + quad
        iW = .5*(iW + iW.T)
        W = la.inv(iW)

        Lambda = np.atleast_2d(wishart.rvs(df=nu,scale=W) )
        self._parameters['Q'] = la.inv(Lambda)

    def _sample_alpha(self):
        if self.hyperparameter_sampling is False:
            return 0
        a0, b0 = 1,1e-1
        a = a0 + self.output_dim / 2
        b = b0 + 1/2*(self.A ** 2.0).sum(0)
        self._parameters['alpha'] = np.atleast_1d(gamma.rvs(a,scale=1.0/b))

    def add_data(self,y,x):
        if x.ndim != 2:
            raise ValueError('input must be 2d')
        if y.ndim != 2:
            raise ValueError('output must be 2d')
        if x.ndim != y.ndim:
            raise ValueError("# of dims of output must equal input")
        if x.shape[0] != y.shape[0]:
            raise ValueError("First dim of output and input must be equal")

        self.x = x.copy()
        self.y = y.copy()
        self.input_dim = x.shape[-1]
        self.N, self.output_dim = y.shape

    def fit(self,y,x,samples=10):
        self.add_data(y=y,x=x)
        super().fit(samples)
        


class BLDS(Gibbs):
    r'''
        Bayesian linear dynamical system.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,state_dim=2,parameter_sampling=True,hyperparameter_sampling=True,system_covariance=True):
        super(BLDS,self).__init__()
        self._dimy = output_dim
        self._dimx = state_dim
        self.parameter_sampling = parameter_sampling
        self.hyperparameter_sampling = hyperparameter_sampling
        self.system_cov = system_covariance

        self.register_parameter("x",None)
        self.initialize()

    @property
    def output_dim(self):
        return self._dimy
    @output_dim.setter
    def output_dim(self,value):
        value = np.maximum(1,value)
        if value != self._dimy:
            self._dimy = value
            self.initialize_output_model()

    @property
    def state_dim(self):
        return self._dimx
    @state_dim.setter
    def state_dim(self,value):
        value = np.maximum(1,value)
        if value != self._dimx:
            self._dimx = value
            self.initialize_state_model()
            self.initialize_prior_model()
    @property
    def output(self):
        y_hat = np.zeros((self.T,self.output_dim))
        for t in range(self.T):
            y_hat[t] = self._emission(t)
        return y_hat

    def initialize(self):
        self.initialize_system_model()
        self.initialize_output_model()

    def initialize_system_model(self):
        A = np.eye(self.state_dim)
        Q = np.eye(self.state_dim)
        m0 = np.random.normal(0,1,self.state_dim)

        self.register_parameter("A",A.copy())
        self.register_parameter("Q",Q.copy())

        self.register_parameter("m0",m0.copy())
        self.register_parameter("P0",Q.copy())
        self.I = np.eye(self.state_dim)

        self.register_parameter('alpha',1e-1*np.ones((self.state_dim)))
        self.register_parameter('beta',1e-1*np.ones(self.state_dim))

    def initialize_output_model(self):

        self.nu0 = self.output_dim+1
        self.iW0 = np.eye(self.output_dim)
        Sigma0 = np.eye(self.output_dim)

        C0 = np.random.uniform(-1,1,(self.output_dim,self.state_dim))
        self.register_parameter("C",C0)
        self.register_parameter("R",Sigma0)

    def _sample_A(self):
        if self.parameter_sampling is False:
            return 0

        Qi = la.inv(self.Q)
        Alpha = np.diag(self.alpha)
        for row in range(self.state_dim):
            ell = Qi[row,row] * (self.x[1:,row] @ self.x[:-1])
            Lam = Qi[row,row] * (self.x[:-1].T @ self.x[:-1]) + Alpha
            Sigma = la.inv(Lam)
            mu = Sigma @ ell
            self._parameters['A'][row] = mvn.rvs(mu,Sigma)

    def _sample_C(self):
        if self.parameter_sampling is False:
            return 0

        Ri = la.inv(self.R)
        Beta = np.diag(self.beta)
        for row in range(self.output_dim):
            ell, Lam = 0,0
            for m in range(self.y.shape[1]):
                t_ = self.delta[:,m]
                ell += Ri[row,row] * (self.y[t_,m,row].T @ self.x[t_])
                Lam += Ri[row,row] * (self.x[t_].T @ self.x[t_])
            Sigma = la.inv(Lam + Beta)
            mu = Sigma @ ell
            self._parameters['C'][row] = mvn.rvs(mu,Sigma)

    def _sample_alpha(self):
        if self.hyperparameter_sampling is False:
            return 0
        a0, b0 = 1,1e-1
        a = a0 + self.state_dim / 2
        b = b0 + 1/2*(self.A ** 2.0).sum(0)
        self._parameters['alpha'] = gamma.rvs(a,scale=1.0/b)

    def _sample_beta(self):
        if self.hyperparameter_sampling is False:
            return 0
        a0, b0 = 1,1e-1
        a = a0 + self.output_dim / 2
        b = b0 + 1/2*(self.C ** 2.0).sum(0)
        self._parameters['beta'] = gamma.rvs(a,scale=1.0/b)

    def _sample_R(self):
        if self.parameter_sampling is False:
            return 0
       
        y_hat = np.expand_dims(self.x @ self.C.T,1)
        x_eps = (self.y - y_hat).reshape(-1,y_hat.shape[-1])
        x_eps = x_eps[self.delta.ravel()]

        Nk = x_eps.shape[0]
        nu = self.nu0 + Nk

        quad = x_eps.T @ x_eps
        iW = self.iW0 + quad
        iW = .5*(iW + iW.T)
        W = la.inv(iW)

        Lambda = np.atleast_2d(wishart.rvs(df=nu,scale=W) )
        self._parameters['R'] = la.inv(Lambda)

    def _sample_Q(self):
        if self.parameter_sampling is False:
            return 0

        if self.system_cov is True:
            nu0 = self.state_dim + 1.0
            iW0 = np.eye(self.state_dim)*1e-1

            Nk = self.T-1
            nu = nu0 + Nk

            x_eps = self.x[1:] - self.x[:-1] @ self.A.T
            iW = iW0 + x_eps.T @ x_eps
            iW = 0.5*(iW + iW.T)
            W = la.inv(iW)

            Lambda = np.atleast_2d(wishart.rvs(df=nu,scale=W) )
            self._parameters['Q'] = la.inv(Lambda)

    def _sample_m0(self):
        if self.parameter_sampling is False:
            return 0

        P0i = la.inv(self.P0)
        Lam0 = np.eye(self.state_dim)
        ell = P0i @ self.x[0]
        Lam = P0i + Lam0
        Sigma = la.inv(Lam)
        mu = Sigma @ ell
        self._parameters['m0'] = mvn.rvs(mu,Sigma)

    def _sample_P0(self):
        if self.parameter_sampling is False:
            return 0

        nu0 = self.state_dim + 1.0
        iW0 = np.eye(self.state_dim)*1e-1

        Nk = 1.0
        nu = nu0 + Nk

        x_eps = self.x[0] - self.m0
        iW = iW0 + x_eps[:,None] @ x_eps[None,:]
        iW = .5*(iW + iW.T)
        W = la.inv(iW)

        Lambda = np.atleast_2d(wishart.rvs(df=nu,scale=W) )
        self._parameters['P0'] = la.inv(Lambda)

    def transition(self,z):
        return self.A @ z
        
    def emission(self,z):
        return self.C @ z
    
    def predict(self,mu,V):
        m = self.transition(mu)
        P = self.A @ V @ self.A.conj().T+ self.Q
        return m, (P)

    def update(self,y,m,P):
        if (y.ndim == 1):
            return self.update_one(y,m,P)
        else:
            return self.update_multiple(y,m,P)

    def update_single(self,y,m,P):
        y_hat = self.emission(m)
        Sigma_hat = self.C @ P @ self.C.conj().T + self.R
        K = la.solve(Sigma_hat, self.C @ P).conj().T
        mu = m + K @ (y - y_hat)
        V = (self.I - K @ self.C) @ P
        return mu, (V)

    def update_multiple(self,y,m,P):
        N = y.shape[0]
        CR = self.C.T @ la.inv(self.R)
        CRC = CR @ self.C

        Lam = la.inv(P)
        ell = Lam @ m
        ell += CR @ y.sum(0)
        Lam += CRC * N
        V = la.inv(Lam)
        mu = V @ ell
        return mu, (V)

    def _forward(self):
        mu = np.zeros((self.T,self.state_dim))
        V = np.zeros((self.T,self.state_dim,self.state_dim))
        m = self.m0.copy()
        P = self.P0.copy()
        for n in range(self.T):
            ''' update'''
            mu[n], V[n] = self.update(self.y[n,self.delta[n]],m,P)
            ''' predict '''
            m,P = self.predict(mu[n], V[n])
        return mu, V

    def _sample_x(self):
        mu, V = self._forward()
        self._parameters['x'][-1] = mvn.rvs(mu[-1],V[-1])
        for t in range(self.T-2,-1,-1):
            m = self.A @ mu[t]
            P = self.A @ V[t] @ self.A.T + self.Q
            K_star = V[t] @ self.A.T @ la.inv(P)

            mu[t] = mu[t] + K_star @ (self.x[t+1] - m)
            V[t] = (self.I - K_star @ self.A) @ V[t]
            self._parameters['x'][t] = mvn.rvs(mu[t],V[t])

    def add_data(self,y,delta=None):
        self.y = y.copy()
        self.T, self.N, self.output_dim = y.shape

        if delta is None:
            delta = np.ones((self.T,self.N))
        if delta.ndim == 1:
            delta = delta[:,None] + np.zeros((1,self.N))
        self.delta = delta.astype(bool).copy()

    def init_samples(self):
        if self._parameters["x"] is None:
            self._parameters["x"] = np.random.multivariate_normal(self.m0, self.P0, self.T)

    def fit(self,y,delta=None,samples=10):
        self.add_data(y,delta=delta)
        self.init_samples()
        super().fit(samples)

    def generate(self,T):
        y = np.zeros((T,self.output_dim))
        y_clean = y.copy()
        x = np.zeros((T,self.state_dim))
        for n in range(T):
            if n == 0:
                x[n] = mvn.rvs(self.m0,self.P0)
            else:
                x[n] = mvn.rvs(self.transition(x[n-1]), self.Q)
            y_clean[n] = self.emission(x[n])
            y[n] = mvn.rvs(y_clean[n], self.R)
        return y, y_clean, x

    def plot_samples(self, params=None):
        if params in self._samples:
            k = list(params)
        else:
            k = self._samples.keys()
        n = len(k)
        fig,ax = plt.subplots(n,figsize=(5,n*1.5))
        ax = np.atleast_1d(ax)
        for j,s in enumerate(k):
            stacked = np.stack(self._samples[s],0)
            if s == 'x':
                ax[j].plot(stacked.transpose(1,0,2)[:,:,0],'b',alpha=.1)
                ax[j].plot(self._estimates['x'][:,0],'k')
            else:
                ax[j].plot(stacked.reshape(stacked.shape[0],-1),'k',alpha=.1,linewidth=1)
            ax[j].set_title(s)
        ax[-1].set_xlabel('step')
        plt.tight_layout()
        






class BLDS2(Gibbs):
    r'''
        Bayesian linear dynamical system.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self,output_dim=1,state_dim=2,switch_dim=1,parameter_sampling=True,hyperparameter_sampling=True,system_covariance=True):
        super(BLDS2,self).__init__()
        self._dimy = output_dim
        self._dimx = state_dim
        self._dimz = switch_dim
        self.parameter_sampling = parameter_sampling
        self.hyperparameter_sampling = hyperparameter_sampling
        self.system_cov = system_covariance

        self.I = np.eye(self.state_dim)

        self.register_parameter("x",None)

        kwds_param = dict(parameter_sampling=self.parameter_sampling,hyperparameter_sampling=self.hyperparameter_sampling)
        
        self.mod_out = NormalWishParams(output_dim=self.output_dim,input_dim=self.state_dim,**kwds_param)
        self.mod_sys = NormalWishParams(output_dim=self.state_dim,input_dim=self.state_dim,**kwds_param)
        self.mod_prior = NormalWishParams(output_dim=self.state_dim,input_dim=1,**kwds_param)

    @property
    def output_dim(self):
        return self._dimy
    @output_dim.setter
    def output_dim(self,value):
        value = np.maximum(1,value)
        if value != self._dimy:
            self._dimy = value

    @property
    def state_dim(self):
        return self._dimx
    @state_dim.setter
    def state_dim(self,value):
        value = np.maximum(1,value)
        if value != self._dimx:
            self._dimx = value

    @property
    def switch_dim(self):
        return self._dimz
    @switch_dim.setter
    def switch_dim(self,value):
        value = np.maximum(1,value)
        if value != self._dimz:
            self._dimz = value

    @property
    def output(self):
        y_hat = np.zeros((self.T,self.output_dim))
        for t in range(self.T):
            y_hat[t] = self._emission(t)
        return y_hat

    @property
    def A(self):
        return self.mod_sys.A
    @property
    def Q(self):
        return self.mod_sys.Q
    @property
    def C(self):
        return self.mod_out.A
    @property
    def R(self):
        return self.mod_out.Q
    @property
    def m0(self):
        return self.mod_prior.A[:,0]
    @property
    def P0(self):
        return self.mod_prior.Q
    
    def transition(self,z):
        return self.A @ z
        
    def emission(self,z):
        return self.C @ z
    
    def predict(self,mu,V):
        m = self.transition(mu)
        P = self.A @ V @ self.A.conj().T+ self.Q
        return m, (P)

    def update(self,y,m,P):
        if (y.ndim == 1):
            return self.update_single(y,m,P)
        else:
            return self.update_multiple(y,m,P)

    def update_single(self,y,m,P):
        y_hat = self.emission(m)
        Sigma_hat = self.C @ P @ self.C.conj().T + self.R
        K = la.solve(Sigma_hat, self.C @ P).conj().T
        mu = m + K @ (y - y_hat)
        V = (self.I - K @ self.C) @ P
        return mu, (V)

    def update_multiple(self,y,m,P):
        N = y.shape[0]
        CR = self.C.T @ la.inv(self.R)
        CRC = CR @ self.C

        Lam = la.inv(P)
        ell = Lam @ m
        ell += CR @ y.sum(0)
        Lam += CRC * N
        V = la.inv(Lam)
        mu = V @ ell
        return mu, (V)

    def _forward(self):
        mu = np.zeros((self.T,self.state_dim))
        V = np.zeros((self.T,self.state_dim,self.state_dim))
        m = self.m0.copy()
        P = self.P0.copy()
        for n in range(self.T):
            ''' update'''
            mu[n], V[n] = self.update(self.y[n,self.delta[n]],m,P)
            ''' predict '''
            m,P = self.predict(mu[n], V[n])
        return mu, V

    def _sample_x(self):
        mu, V = self._forward()
        self._parameters['x'][-1] = mvn.rvs(mu[-1],V[-1])
        for t in range(self.T-2,-1,-1):
            m = self.A @ mu[t]
            P = self.A @ V[t] @ self.A.T + self.Q
            K_star = V[t] @ self.A.T @ la.inv(P)

            mu[t] = mu[t] + K_star @ (self.x[t+1] - m)
            V[t] = (self.I - K_star @ self.A) @ V[t]
            self._parameters['x'][t] = mvn.rvs(mu[t],V[t])


    def _update_mod_out(self):
        rz,cz = np.nonzero(self.delta)
        _y = self.y[rz,cz]
        _x = self.x[rz]
        self.mod_out.add_data(y=_y,x=_x)

    def _update_mod_sys(self):
        self.mod_sys.add_data(y=self.x[1:],x=self.x[:-1])

    def _update_mod_prior(self):
        _y = np.atleast_2d(self.x[0])
        _x = np.atleast_2d(1.0)
        self.mod_prior.add_data(y=_y,x=_x)

    def add_data(self,y,z=None,delta=None):
        self.y = y.copy()
        self.T, self.N, self.output_dim = y.shape


        if delta is None:
            delta = np.ones((self.T,self.N))
        if delta.ndim == 1:
            delta = delta[:,None] + np.zeros((1,self.N))
        self.delta = delta.astype(bool).copy()

        if z is None:
            z = np.ones(self.T)
        if z.ndim > 1:
            raise ValueError("z must be 1d.")
        if z.max() > self.switch_dim:
            raise ValueError("model's switch dim less than in z")
        self.z = z.copy().astype(int)

    def init_samples(self):
        if self._parameters["x"] is None:
            self._parameters["x"] = mvn.rvs(self.m0, self.P0, self.T)

    def fit(self,y,z=None,delta=None,samples=10,burn_rate=.75):
        self.add_data(y,delta=delta,z=z)
        self.init_samples()
        super().fit(samples=samples,burn_rate=burn_rate)

    def generate(self,T):
        y = np.zeros((T,self.output_dim))
        y_clean = y.copy()
        x = np.zeros((T,self.state_dim))
        for n in range(T):
            if n == 0:
                x[n] = mvn.rvs(self.m0,self.P0)
            else:
                x[n] = mvn.rvs(self.transition(x[n-1]), self.Q)
            y_clean[n] = self.emission(x[n])
            y[n] = mvn.rvs(y_clean[n], self.R)
        return y, y_clean, x

    def plot_samples(self, params=None):
        if params in self._samples:
            k = list(params)
        else:
            k = self._samples.keys()
        n = len(k)
        fig,ax = plt.subplots(n,figsize=(5,n*1.5))
        ax = np.atleast_1d(ax)
        for j,s in enumerate(k):
            stacked = np.stack(self._samples[s],0)
            if s == 'x':
                ax[j].plot(stacked.transpose(1,0,2)[:,:,0],'b',alpha=.1)
                ax[j].plot(self._estimates['x'][:,0],'k')
            else:
                ax[j].plot(stacked.reshape(stacked.shape[0],-1),'k',alpha=.1,linewidth=1)
            ax[j].set_title(s)
        ax[-1].set_xlabel('step')
        plt.tight_layout()

        for k in self._modules:
            self._modules[k].plot_samples()
         