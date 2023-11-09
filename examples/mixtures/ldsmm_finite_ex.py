#%%
from gibbs import Gibbs, get_colors, LDS, Data, Module, HMM
import numpy as np
import matplotlib.pyplot as plt
from sequential.lds import ode_polynomial_predictor
from scipy.stats import multivariate_normal as mvn
from scipy.stats import invgamma

class MixtureLDS(Module):
    """
    One of many -  mixture and discrete source state both encoded by the hmm. Assumes one source, and creates one point out of many for each time. Converges quickly because all discrete states are encoded in one variable, rather than having factorized the mixture and the source's temporal discrete state, which mixes very slowly due to strong correlations between the two variables. 
    """
    def __init__(self,output_dim,state_dim=2,states=2,learn=False):
        super().__init__()

        self.states = states
        N = states - 2
        self.learn = learn

        self.lds = LDS(output_dim=1,state_dim=state_dim,parameter_sampling=False,hyper_sample=False,full_covariance=False)
        self.hmm = HMM(states=states,parameter_sampling=False)


        _C = np.zeros((1,state_dim))
        _C[0,0] = 1.0

        sigma = .05
        sigma_sys = .01

        self.lds.theta[0].sys._parameters['A'] = ode_polynomial_predictor(order=state_dim)
        self.lds.theta[0].sys._parameters['Q'] = np.eye(state_dim)*sigma_sys**2
        self.lds.theta[0].obs._parameters['A'] = _C
        self.lds.theta[0].obs._parameters['Q'] = np.eye(1)*(sigma**2)
        self.lds.theta[0].pri._parameters['A'] = np.zeros(state_dim)[:,None]
        self.lds.theta[0].pri._parameters['Q'] = np.eye(state_dim)*2

        Gam = np.eye(states)
        Gam[0,1:-1] = 1e-1 / N 
        Gam[1:-1,1:-1] = 1 / N
        Gam[1:-1,-1] = 1e-2 / N
        pi = np.zeros(states)
        pi[0] = 1
        pi[1:-1] = 1e-1
        Gam /= Gam.sum(-1)[:,None]
        pi /= pi.sum()

        self.hmm._parameters['Gamma'] = (Gam)
        self.hmm._parameters['pi'] = (pi)
        self.hmm.log_Gamma = np.log(Gam)
        self.hmm.log_pi = np.log(pi)

    def forward(self,data:'Data'):
        T = data.T
        if self.lds.x.shape[0] != T:   
            self.lds.forward(data)

        noise_logpdf = mvn.logpdf(0,0,10**2)
        y_hat = self.lds.x @ self.lds.C(0).T
        logl = np.zeros((T,self.states)) 
        for t in range(T):
            N = (data.time == t).sum()
            logl[t] = noise_logpdf * N
            logl[t,1:-1] = mvn.logpdf(data.output[data.time == t], y_hat[t],self.lds.R(0)) + noise_logpdf*(N-1)
        self.hmm.forward(logl=logl)

        t_on = (self.hmm.z > 0) & (self.hmm.z < self.states-1)
        m_on = self.hmm.z[t_on]-1 + np.arange(T)[t_on]*(self.states-2)
        filt = np.zeros(len(data)).astype(bool)
        filt[m_on] = True
        _data = data.filter(filt)
        self.lds.forward(data=_data)

        if self.learn:
            y_hat = self.lds.x @ self.lds.C(0).T
            y_tilde = _data.output - y_hat[_data.time]
            N = len(_data)
            eps = np.trace(y_tilde.T @ y_tilde)

            a0 = 2
            b0 = a0 * (.01)**2
            a = a0 + 1/2*N
            b = b0 + 1/2*eps
            sigma2 = invgamma.rvs(a=a,scale=b)
            self.lds.theta[0].obs._parameters['Q'] = np.eye(1)*sigma2

            x_tilde = self.lds.x[1:] - self.lds.x[:-1] @ self.lds.A(0).T
            N = x_tilde.size
            eps = np.trace(x_tilde.T @ x_tilde)

            a0 = 2
            b0 = a0 * (.01)**2
            a = a0 + 1/2*N
            b = b0 + 1/2*eps
            sigma2 = invgamma.rvs(a=a,scale=b)
            self.lds.theta[0].sys._parameters['Q'] = np.eye(self.lds.state_dim)*sigma2


#%%

T = 160
N = 2
np.random.seed(123)
time = np.arange(T)

f = .1

y = np.cos(time*f)*.6
y[:30] = np.random.uniform(-1,1,30)
y[-10:] = np.random.uniform(-1,1,10)
y[-30:] = np.random.uniform(-1,1,len(y[-30:]))

noise = np.random.uniform(-1,1,(T,N-1))
if N > 1:
    y = np.concatenate([y[:,None],noise],-1).ravel()
time = np.stack([time]*N,-1).ravel()
y += np.random.normal(0,1e-3,y.shape)

data = Data(y=y[:,None],time=time)
data.plot()

#%%
model = MixtureLDS(output_dim=1,states=2+N,learn=True)
sampler = Gibbs()

#%%
sampler.fit(data=data,model=model,samples=50)

# %%
sampler.get_estimates(burn_rate=.75)
z_hat = sampler._estimates['hmm.z']
x_hat = sampler._estimates['lds.x']
y_hat = x_hat @ model.lds.C(0).T
chain = sampler.get_chain()
y_chain = chain['lds.x'] @ model.lds.C(0).T

colors = get_colors()
plt.figure(figsize=(3,2))
plt.imshow(colors[chain['hmm.z']])
plt.xlabel('data point')
plt.ylabel('sample')
plt.tight_layout()
# plt.savefig('imgs/ldsmm_finite_zchain_ex.pdf')

# %%
t_on = (z_hat > 0) & (z_hat < model.states-1)
m_on = z_hat[t_on]-1 + np.arange(T)[t_on]*(model.states-2)
filt = np.zeros(len(data)).astype(int)
filt[m_on] = 1

plt.figure(figsize=(3,2))
plt.plot(y_chain[:,:,0].T,color='b',alpha=5/y_chain.shape[0],zorder=1)
plt.plot(y_hat,color='b')
plt.scatter(data.time,data.output,color=colors[filt],linewidth=0,s=5)
plt.xlabel('time')
plt.ylim(data.output.min()-.2,data.output.max()+.2)
plt.xlim(0,data.T-1)
plt.tight_layout()
plt.savefig('imgs/ldsmm_finite_ex.pdf')

# %%
# plt.plot(chain['lds.theta.0.obs.Q'][:,0,0],'b')
# plt.plot(chain['lds.theta.0.sys.Q'][:,0,0],'r')
# plt.xlim(0)
# plt.ylim(0)
# # %%

# %%
