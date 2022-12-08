
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