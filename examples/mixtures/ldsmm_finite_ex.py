#%%
from gibbs import Gibbs, lds_test_data, tqdm, get_scatter_kwds, get_colors, LDS, Data, Mixture, Plate, Module, SLDS, HMM
import numpy as np
import matplotlib.pyplot as plt
import os
from sequential.lds import ode_polynomial_predictor
from scipy.stats import multivariate_normal as mvn

# TODO - on/off
plt.style.use('sines-latex')


class MixtureLDS(Module):
    def __init__(self,output_dim,state_dim=2,states=2):
        super().__init__()

        self.states = states
        N = states - 2

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
        logl = np.zeros((T,self.states)) 
        if self.hmm.z.shape[0] != T:   
            self.hmm.forward(logl=logl)

        t_on = (self.hmm.z > 0) & (self.hmm.z < self.states-1)
        m_on = self.hmm.z[t_on]-1 + np.arange(T)[t_on]*(self.states-2)
        filt = np.zeros(len(data)).astype(bool)
        filt[m_on] = True
        self.lds.forward(data=data.filter(filt))

        noise_logpdf = mvn.logpdf(0,0,10**2)
        y_hat = self.lds.x @ self.lds.C(0).T
        for t in range(T):
            logl[t] = noise_logpdf * (self.states-2)
            logl[t,1:-1] = mvn.logpdf(data.output[data.time == t], y_hat[t],self.lds.R(0)) + noise_logpdf*(self.states-2-1)
        self.hmm.forward(logl=logl)

#%%
T = 160
N = 3
np.random.seed(123)
time = np.arange(T)

f = .1

y = np.cos(time*f)
y[:20] = np.random.uniform(-1,1,20)
y[-10:] = np.random.uniform(-1,1,10)
y[T//2:T//2+20] = np.random.uniform(-1,1,20)

noise = np.random.uniform(-1,1,(T,N-1))
if N > 1:
    y = np.concatenate([y[:,None],noise],-1).ravel()
time = np.stack([time]*N,-1).ravel()
y += np.random.normal(0,1e-3,y.shape)

data = Data(y=y[:,None],time=time)
data.plot()

#%%
model = MixtureLDS(output_dim=1,states=2+N)
sampler = Gibbs()

#%%
sampler.fit(data=data,model=model,samples=10*N)


# %%
sampler.get_estimates(burn_rate=.75)
z_hat = sampler._estimates['z']
x_hat = sampler._estimates['x']
y_hat = x_hat @ lds.C(0).T
chain = sampler.get_chain()
y_chain = chain['x'] @ lds.C(0).T

colors = get_colors()
plt.figure(figsize=(3,2))
plt.imshow(colors[chain['z']])
plt.xlabel('data point')
plt.ylabel('sample')
plt.tight_layout()
plt.savefig('imgs/ldsmm_finite_zchain_ex.pdf')

# %%
t_on = (z_hat > 0) & (z_hat < states-1)
m_on = z_hat[t_on]-1 + np.arange(T)[t_on]*N
filt = np.zeros(len(data)).astype(int)
filt[m_on] = 1

plt.figure(figsize=(3,2))
plt.plot(y_chain[:,:,0].T,color=colors[1],alpha=.1)
plt.plot(y_hat,color=colors[1])
plt.scatter(data.time,data.output,color=colors[filt],linewidth=0,s=10)
plt.xlabel('time')
plt.ylim(data.output.min()-.2,data.output.max()+.2)
plt.tight_layout()
plt.savefig('imgs/ldsmm_finite_ex.pdf')

# %%
