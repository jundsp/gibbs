#%%
from gibbs import Gibbs, get_colors, LDS, Data, Module, HMM, categorical2multinomial
import numpy as np
import matplotlib.pyplot as plt
import os
from sequential.lds import ode_polynomial_predictor
from scipy.stats import multivariate_normal as mvn
from scipy.stats import invgamma
from itertools import product

class MixtureLDS(Module):
    """
    many of many -  mixture and discrete source state both encoded by the hmm. Assumes one source and one noise, that creates multiple point out of many for each time. Converges quickly because all discrete states are encoded in one variable, rather than having factorized the mixture and the source's temporal discrete state, which mixes very slowly due to strong correlations between the two variables. 
    """
    def __init__(self,output_dim,state_dim=2,states=2,learn=False,N=1):
        super().__init__()

        Comb = product(np.arange(2), repeat=N)
        Comb = np.array(list(Comb))
        Comb = np.concatenate([Comb,Comb[[0]]],0)
        self.Comb = Comb + 0
        states = Comb.shape[0]

        self.states = states
        self.learn = learn

        self.lds = LDS(output_dim=1,state_dim=state_dim,parameter_sampling=False,hyper_sample=False,full_covariance=False)
        self.hmm = HMM(states=states,parameter_sampling=False)

        sigma = .1
        sigma_sys = .05

        _C = np.zeros((1,state_dim))
        _C[0,0] = 1.0

        self.lds.theta[0].sys._parameters['A'] = ode_polynomial_predictor(order=state_dim)
        self.lds.theta[0].sys._parameters['Q'] = np.eye(state_dim)*sigma_sys**2
        self.lds.theta[0].obs._parameters['A'] = _C
        self.lds.theta[0].obs._parameters['Q'] = np.eye(1)*(sigma**2)
        self.lds.theta[0].pri._parameters['A'] = np.zeros(state_dim)[:,None]
        self.lds.theta[0].pri._parameters['Q'] = np.eye(state_dim)

        states_on = states - 2
        Gam = np.eye(states)
        Gam[0,1] = 1e-1 / states_on
        Gam[1:-1,1:-1] = 1 / states_on
        Gam[1:-1,-1] = 1e-3 / states_on
        pi = np.zeros(states)
        pi[0] = 1
        pi[1:-1] = 1e-1 / states_on
        Gam /= Gam.sum(-1)[:,None]
        pi /= pi.sum()

        self.hmm.set_parameters(Gamma=Gam,pi=pi)

        self._parameters['sigma2_noise'] = np.atleast_1d(1**2)

    def forward(self,data:'Data'):
        T = data.T
        if self.lds.x.shape[0] != T:   
            self.lds.forward(data)

        y_hat = self.lds.x @ self.lds.C(0).T
        logl = np.zeros((T,self.states)) 
        
        for t in range(T):
            m_on = np.nonzero(data.time==t)[0]
            N = len(m_on)
            temp = np.zeros((N,2))
            temp[:,0] = mvn.logpdf(data.output[m_on]*0, 0, np.eye(1)*self.sigma2_noise)
            temp[:,1] = mvn.logpdf(data.output[m_on], y_hat[t],self.lds.R(0))
            for k in range(model.hmm.states):
                logl[t,k] = temp[np.arange(N),self.Comb[k]].sum()

        self.hmm.forward(logl=logl)

        _data = data.filter(self.Comb[self.hmm.z].ravel() == 1)
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

            _data_noise = data.filter(self.Comb[self.hmm.z].ravel() == 0)
            a = 1 + 1/2*len(_data_noise)
            b = 1 + 1/2*np.trace(_data_noise.output.T @ _data_noise.output)
            self._parameters['sigma2_noise'] = invgamma.rvs(a=a,scale=b)


#%%

T = 80
N = 5
np.random.seed(123)

time = np.arange(T)

f = .13

y = np.cos(time*f)*.6
y = np.stack([y]*4,-1)
y[:15] = np.random.uniform(-1,1,y[:15].shape)
y[-20:] = np.random.uniform(-1,1,y[-20:].shape)

noise = np.random.uniform(-1,1,(T,N-1))
if N > 1:
    y = np.concatenate([y,noise],-1)
N = y.shape[-1]
y = np.sort(y,-1)
y = y.ravel()
time = np.stack([time]*N,-1).ravel()
y += np.random.normal(0,1e-2,y.shape)

data = Data(y=y[:,None],time=time)
data.plot()

#%%
model = MixtureLDS(output_dim=1,N=N,learn=True)
sampler = Gibbs()

#%%
sampler.fit(data=data,model=model,samples=30)

# %%
chain = sampler.get_chain(burn_rate=0)

z_hat = categorical2multinomial(chain['hmm.z']).mean(0).argmax(-1)
x_hat = sampler._estimates['lds.x']
y_hat = x_hat @ model.lds.C(0).T

y_chain = chain['lds.x'] @ model.lds.C(0).T

colors = get_colors()
plt.figure(figsize=(3,2))
plt.imshow(chain['hmm.z'])
plt.xlabel('data point')
plt.ylabel('sample')
plt.tight_layout()
plt.savefig('imgs/ldsmm_finite_zchain_ex.pdf')

# %%
filt = model.Comb[z_hat].ravel()

plt.figure(figsize=(4,2.5))
plt.plot(y_chain[:,:,0].T,color='b',alpha=5/y_chain.shape[0],zorder=1)
plt.plot(y_hat,color='b')
plt.scatter(data.time,data.output,color=colors[filt],linewidth=0,s=10,zorder=2)
plt.xlabel('time')
plt.ylim(data.output.min()-.2,data.output.max()+.2)
plt.xlim(0,T-1)
plt.tight_layout()
plt.savefig('imgs/ldsmm_finite_ex.pdf')

# %%
plt.plot(chain['lds.theta.0.obs.Q'][:,0,0]**.5,'b')
plt.plot(chain['lds.theta.0.sys.Q'][:,0,0]**.5,'r')
plt.plot(chain['sigma2_noise'].ravel()**.5,'g')
plt.xlim(0)
plt.ylim(0)

# %%
