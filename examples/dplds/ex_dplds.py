#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_t
from scipy.special import logsumexp
import matplotlib as mpl
import sequential.lds as lds
import scipy.linalg as la
from numpy.random import permutation as randperm
mpl.rcParams['figure.dpi'] = 150

from gibbs import GibbsDirichletProcess

'''
Infinite mixture of linear dynamical systems.

Dirichlet process, gibbs sampling. 

© Julian Neri, 2022
'''

class DP_LDS(GibbsDirichletProcess):
    def __init__(self,output_dim=1,alpha=1,order=2,tau=.1,state_dim=1,learn=True,emission="lds",dynamics="lds"):
        super().__init__(alpha=alpha,learn=learn)
        self.output_dim = output_dim
        self.order = order
        self.state_dim = state_dim
        self.tau = tau
        self._kinds = []
        self._dynamics = dynamics
        self._emission = emission

    @property
    def latent_dim(self):
        return self.order * self.state_dim

    def _block_state_moments(self):
        if self._dynamics == "lds":
            return self.lds_block_state()
        else: 
            return self.gp_state()

    def gp_state(self):
        '''
        Gaussian process prior with RBF / linear kernel
        '''
        self.order = 1
        block_dim = self.T * self.latent_dim
        t = np.linspace(-1,1,block_dim)[:,None]
        b0,b1,b2 = .01, 10, 10
        Sigma = b0 * np.eye(block_dim) + b1*(t @ t.T) +  b2* np.exp(-(t-t.T)**2.0 / .1)
        ell = np.zeros(block_dim)
        return ell, la.inv(Sigma)

    def lds_block_state(self):
        '''
        Linear dynamical system, block prior 
        '''
        taus = np.ones(self.latent_dim) * self.tau**2.0
        taus[self.order*(self.state_dim-1):] *= 1e-1
        Q = np.diag(taus)
        
        system_mtx = lds.polynomial_matrix(self.order,shift=1)
        A = np.kron(np.eye(self.state_dim),system_mtx)
        P0 = np.eye(self.latent_dim)*100
        m0 = np.zeros(self.latent_dim)

        Qi = la.inv(Q)
        Pi = la.inv(P0)

        Omega = Qi @ A
        AQA = Omega.T @ A
        Psi_t = Qi + AQA
        Psi_1 = Pi + AQA
        Psi_T = Qi
        # Construct block precision matrix
        Lambda = np.kron(np.eye(self.T),Psi_t)
        Lambda[:self.latent_dim,:self.latent_dim] = Psi_1
        Lambda[-self.latent_dim:,-self.latent_dim:] = Psi_T
        Lambda[self.latent_dim:,:-self.latent_dim] += np.kron(np.eye(self.T-1),-Omega)
        Lambda[:-self.latent_dim,self.latent_dim:] += np.kron(np.eye(self.T-1),-Omega.T)

        ell = np.zeros(self.T*self.latent_dim)
        ell[:self.latent_dim] = Pi @ m0
        
        return ell, Lambda

    def _block_output_moments(self):
        return self.emission_block_output()

    def emission_block_output(self):
        self.C = np.zeros((self.y.shape[0],self.latent_dim*self.T))
        for t in range(self.T):
            n = np.nonzero(t == self.t)[-1]
            if len(n) > 0:
                idx = self._n2idx(n)
                self.C[idx,t*self.latent_dim:(t+1)*self.latent_dim] = self.emission_fun(self.x[n])
                
        Lambda = self.C.T @ self.C
        ell = self.C.T @ self.y
        return ell, Lambda

    def emission_fun(self,x):
        x = np.atleast_1d(x).reshape(-1,1)
        m = np.arange(self.state_dim-self.output_dim+1)[None,:]
        _C = np.zeros((x.shape[0]*self.output_dim,self.latent_dim))
        if self._emission == "cepstrum":
            _C[::self.output_dim,::self.order] = np.cos(np.pi*x*m)
        elif self._emission == "polynomial":
            _C[::self.output_dim,:self.order*(self.state_dim-self.output_dim+1):self.order] = np.power(x, m)
        else:
            _C[0] = 1
        for j in range(1,self.output_dim):
            _C[j::self.output_dim,self.order*(self.state_dim-self.output_dim+j)::self.latent_dim] = 1
        return _C

    def _init_moments(self):
        self.theta_prior = [[],[]]

        self.ell_state, self.Lambda_state = self._block_state_moments()
        Lambda = self.Lambda_state.copy()
        V = la.inv(self.Lambda_state)
        mu = V @ self.ell_state
        tau_ev = 100.0
        a = 3
        b = a / tau_ev
        self.theta_prior[0] = dict(m=mu,V=V,Lambda=Lambda,a=a,b=b)

        mu_noise = np.zeros(mu.shape)
        V_noise = np.eye(V.shape[0])
        tau_ev = .1
        a = 3
        b = a / tau_ev
        self.theta_prior[1] = dict(m=mu_noise,V=V_noise,Lambda=la.inv(V_noise),a=a,b=b)

        self.ell_output, self.Lambda_output = self._block_output_moments()

    def _prior_predictive(self,n,kind=0):
        return self._predictive(n,self.theta_prior[kind])
    
    def _posterior_predictive(self,n,idx,kind):
        theta_post = self._posterior(idx,kind=kind)
        return self._predictive(n,theta_post)

    def _n2idx(self,n):
        return (np.atleast_1d(n)[:,None]*self.output_dim + np.arange(self.output_dim)[None,:]).ravel()

    def _predictive(self,n,theta):
        mu_y,Sigma_y,nu_y = self._predictive_parameters(self.t[n],self.x[n],theta)
        idx = self._n2idx(n)
        return multivariate_t.logpdf(self.y[idx],loc=mu_y,shape=Sigma_y,df=nu_y)

    # Multivariate t
    def _predictive_parameters(self,t,x,theta):
        m,V,a,b = theta['m'], theta['V'], theta['a'], theta['b']
        C_ = np.atleast_2d(self.emission_fun(x))
        j = t*self.latent_dim
        k = j + self.latent_dim
        m_ = m[j:k]
        V_ = V[j:k,j:k]

        mu_y = C_ @ m_
        I = np.eye(C_.shape[0])
        Sigma_y = b/a * (C_ @ V_ @ C_.T + I)
        nu_y = 2*a

        return mu_y, Sigma_y, nu_y
        
    def _posterior(self,n,kind):
        N = len(n)
        _theta = self.theta_prior[kind]
            
        m0,Lam0,a0,b0 = _theta['m'], _theta['Lambda'], _theta['a'], _theta['b']

        # idx spans N * M 
        idx = self._n2idx(n)

        ell_hat = self.ell_state + self.C[idx].T @ self.y[idx]
        Lam_hat = self.Lambda_state + self.C[idx].T @ self.C[idx]

        # Main cost is here. inversion of TD X TD matrix
        V = la.inv(Lam_hat)
        mu = V @ ell_hat

        a_hat = a0 + 0.5*len(idx)
        b_hat = b0 + 0.5*(self.y[idx].T @ self.y[idx] + m0.T @ Lam0 @ m0 - mu.T @ Lam_hat @ mu)

        return dict(m=mu,V=V,Lam=Lam_hat,a=a_hat,b=b_hat)
    
    def _sample_one_z(self,n):
        rho = np.zeros(self.K+2) - np.inf
        for k in range(self.K):
            kind = self._kinds[k]
            idx = self._parameters['z'] == k
            idx[n] = False
            idx = np.nonzero(idx)[-1]
            if len(idx) > 0:
                rho[k] = self._posterior_predictive(n,idx,kind=kind) + np.log(len(idx))

        rho[-2] = self._prior_predictive(n,kind=0) + np.log(self._parameters['alpha'])
        rho[-1] = self._prior_predictive(n,kind=1) + np.log(self._parameters['alpha'])

        rho -= logsumexp(rho)
        rho = np.exp(rho)
        rho /= rho.sum()
        _z_now = np.random.multinomial(1,rho).argmax()
        
        if _z_now > (self.K-1):
            if _z_now == self.K:
                kind = 0
            else:
                kind = 1
            _z_now = self.K
            self._kinds.append(kind)
            self.K += 1
        self._parameters['z'][n] = _z_now

    def _sample_z(self):
        for n in randperm(self.N):
            self._sample_one_z(n)
        self._collapse_groups()

    def fit(self,t,x,y,samples=100):
        self.T = t.max()+1
        self.t = t.copy()
        self.x = x.copy()
        self.is3d = len(np.unique(x)) > 1
        if y.ndim == 1: y = y.reshape(-1,1)
        N, dimx = y.shape
        self.y = y.ravel()
        if (dimx != self.output_dim) | ("N" not in self.__dict__):
            self.output_dim = dimx
            self.N = N
            self._init_moments()

        if self._parameters['z'] is None:
            self._parameters['z'] = np.zeros(self.N).astype(int)-1
            self.K = 0

        super().fit(samples)

    def plot(self):
        if self.is3d:
            self._plot_3d()
        else:
            self._plot_2d()

    def _plot_2d(self):
        z_hat = self._parameters['z'].astype(int)
        K_hat = len(np.unique(z_hat))
        colors = np.array(['r','g','b','m','y','k','orange']*30)
        plt.figure(figsize=(5,4))
        plt.scatter(self.t,self.y[::self.output_dim],c=colors[z_hat])

        sigma_y = np.zeros((self.T))
        mu_y = np.zeros((self.T))

        for k in range(K_hat):
            idx = z_hat == k
            kind = self._kinds[k]
            theta = model._posterior(idx,kind=kind)
            for t in range(self.T):
                mu_y[t],sigma_y[t],nu_y = model._predictive_parameters(t,x=0,theta=theta)
                sigma_y[t] *= nu_y/(nu_y-2.0)
            plt.plot(mu_y,color=colors[k])
            plt.fill_between(x=np.arange(self.T),y1=mu_y + sigma_y**.5,y2=mu_y-sigma_y**.5,alpha=.2,color=colors[k],linewidth=0)
        plt.ylim(self.y.min(),self.y.max())

    def _plot_3d(self):
        z_hat = self._parameters['z'].astype(int)
        K_hat = len(np.unique(z_hat))
        colors = np.array(['r','g','b','m','y','k','orange']*30)
        nx = 32 
        sigma_y = np.zeros((K_hat,self.T,nx,self.output_dim))
        mu_y = np.zeros((K_hat,self.T,nx,self.output_dim))
        _x_plot = np.linspace(self.x.min(),self.x.max(),nx)
        X, T = np.meshgrid(_x_plot,np.arange(self.T))
        for k in range(K_hat):
            idx = np.nonzero(z_hat == k)[-1]
            kind = self._kinds[k]
            theta = model._posterior(idx,kind=kind)
            for j,x in enumerate(_x_plot):
                for t in range(self.T):
                    mu_Y,sigma_Y,nu_Y= model._predictive_parameters(t,x=x,theta=theta)
                    sigma_Y *= nu_Y/(nu_Y-2.0)
                    for m in range(self.output_dim):
                        mu_y[k,t,j,m] = mu_Y[m]
                        sigma_y[k,t,j,m] = sigma_Y[m,m]**.5

        for m in range(self.output_dim):
            fig = plt.figure(figsize=(5,4))
            ax = fig.add_subplot(projection='3d',proj_type="ortho")
            ax.scatter(self.t,self.x,self.y[m::self.output_dim],c=colors[z_hat],s=15,linewidth=0)
            for k in range(K_hat):
                ax.plot_surface(T, X, mu_y[k,:,:,m],alpha=.1,linewidth=0,color=colors[k],antialiased=False)
                ax.set_zlim(self.y.min(),self.y.max())
                ax.set_xlabel('t')
                ax.set_ylabel('x')
                ax.set_zlabel('y')

        

#%%
# np.random.seed(123)
T = 10
M = 20
K = 2
_t, _x, _y = [], [], []
y = np.sin(2*np.pi*(np.arange(T)-T/2)/T*.5)
y = np.stack([y,-y],-1)
for t in range(T):
    for m in range(M):
        for k in range(K):
            _t.append(t)
            _x.append(m)
            _y.append(y[t,k])

t, x, y = np.array(_t), np.array(_x), np.array(_y)

y += np.random.normal(0,.2,y.shape)
y = np.stack([y,-y*0],-1)
x = (x-x.min())/(x.max()-x.min())*2-1

fig,ax = plt.subplots(2,subplot_kw=dict(projection="3d",proj_type="ortho"))
ax[0].scatter3D(t,x,y[:,0],s=10,color='k')
ax[1].scatter3D(t,x,y[:,1],s=10,color='k')

model = DP_LDS(order=2,state_dim=3,tau=.5,emission="polynomial")

#%%
model.fit(t=t,y=y,x=x,samples=10)
model.plot()
model.plot_samples()

# %%
