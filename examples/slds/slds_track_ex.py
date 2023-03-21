#%%
from gibbs import Gibbs, LDS, tqdm, get_scatter_kwds, get_colors, Data, HMM, Module
import numpy as np
import matplotlib.pyplot as plt
import os
from sequential.lds import ode_polynomial_predictor, ode_covariance
from scipy.stats import invgamma, multivariate_normal

plt.style.use('sines-latex')

class Tracker(Module):
    def __init__(self,states=1,state_dim=2):
        super().__init__()
        self.states = states
        self.state_dim = state_dim
        self.lds = LDS(output_dim=1,state_dim=state_dim,parameter_sampling=False,hyper_sample=False,full_covariance=False)
        self.hmm = HMM(states=states,parameter_sampling=False,circular=True)

        self.init_lds()

    def init_lds(self):
        sigma = .1
        sigma_sys = .01
        _C = np.zeros((1,self.state_dim))
        _C[0,0] = 1.0

        self.lds.theta[0].sys._parameters['A'] = ode_polynomial_predictor(order=self.state_dim)
        self.lds.theta[0].sys._parameters['Q'] = np.eye(self.state_dim)*sigma_sys**2
        self.lds.theta[0].obs._parameters['A'] = _C
        self.lds.theta[0].obs._parameters['Q'] = np.eye(1)*(sigma**2)
        self.lds.theta[0].pri._parameters['A'] = np.zeros(self.state_dim)[:,None]
        self.lds.theta[0].pri._parameters['Q'] = np.eye(self.state_dim)*2

    def forward(self,data:'Data'):
        T = data.T
        logl = np.zeros((T,self.states))
        if len(self.hmm.z) != data.T:            
            self.hmm.forward(logl=logl)
        _t = np.arange(T)
        m = _t * self.states + self.hmm.z
        self.lds.forward(data=Data(y=data.output[m],time = _t))

        y_hat = self.lds.x @ self.lds.C(0).T
        for t in range(T):
            logl[t] = multivariate_normal.logpdf(data.output[data.time == t], y_hat[t],self.lds.R(0))
        self.hmm.forward(logl=logl)

        m = _t * self.states + self.hmm.z
        y_hat = self.lds.x @ self.lds.C(0).T

        y_tilde = data.output[m] - y_hat
        N = y_tilde.size
        eps = np.trace(y_tilde.T @ y_tilde)

        a0 = 2
        b0 = a0 * (.001)**2
        a = a0 + 1/2*N
        b = b0 + 1/2*eps
        sigma2 = invgamma.rvs(a=a,scale=b)
        self.lds.theta[0].obs._parameters['Q'] = np.eye(1)*sigma2

        x_tilde = self.lds.x[1:] - self.lds.x[:-1] @ self.lds.A(0).T
        N = x_tilde.size
        eps = np.trace(x_tilde.T @ x_tilde)

        a0 = 2
        b0 = a0 * (.001)**2
        a = a0 + 1/2*N
        b = b0 + 1/2*eps
        sigma2 = invgamma.rvs(a=a,scale=b)
        self.lds.theta[0].sys._parameters['Q'] = np.eye(self.state_dim)*sigma2

        self.lds.theta[0].pri.forward(y=self.lds.x[[0]])

#%%
T = 100
M = 5
np.random.seed(123)

fvec = np.zeros(T)
f = 2/T
for t in range(T):
    fvec[t] = f

time = np.arange(T)
fvec = np.sin(np.pi*time/T/2)*.03
fvec = np.linspace(.01,.02,T)
phase = np.cumsum(2*np.pi*fvec)
y = np.cos(phase)*.2+.5

y = y[:,None] + np.arange(M)[None,:]
y[:,1:] = np.random.uniform(0,1,(T,M-1))
y = np.sort(y,-1)
y = y.ravel()[:,None]
time = np.stack([time]*M,-1).ravel()
y += np.random.normal(0,.005,y.shape)
y -= np.mean(y)

data = Data(y=y,time=time)
# data = data.filter((time < T//2) | (time > (T//2+20)))
data.plot()

#%%
model = Tracker(states=M,state_dim=2)
sampler = Gibbs()

#%%
sampler.fit(data=data,model=model,samples=100)

# %%
sampler.get_estimates(burn_rate=.95)
chain = sampler.get_chain(burn_rate=.5)
z_hat = sampler._estimates['hmm.z']
x_hat = sampler._estimates['lds.x']
m = np.arange(data.T) * M + z_hat

plt.figure(figsize=(3,2))
plt.scatter(time,y,s=10,linewidth=0,alpha=.5)
plt.scatter(data.time[m],data.output[m],color='k',s=10,linewidth=0)
plt.plot(chain['lds.x'][:,:,0].T,'r',alpha=.1)
plt.plot(x_hat[:,0],'r')
# plt.ylim(0,1)
plt.tight_layout()
plt.savefig("imgs/slds_track_ex.pdf")

# %%
plt.imshow(chain['hmm.z'])

# %%
plt.plot(chain['lds.theta.0.sys.Q'][:,0,0])
plt.plot(chain['lds.theta.0.obs.Q'][:,0,0])

# %%
