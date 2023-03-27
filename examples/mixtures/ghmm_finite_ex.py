#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma, norm, dirichlet, multivariate_normal as mvn
from itertools import product

from gibbs import Gibbs, get_colors, Data, Mixture, Module, HMM, logsumexp, categorical2multinomial, NormalWishart, Plate


plt.style.use('sines-latex')

def make_big_hmm(Gam,pi,components):
    lookup = np.array(list(product(np.arange(Gam.shape[0]), repeat=components-1))).astype(int).T

    states = lookup.shape[-1]
    pi_full = np.zeros(states)
    Gam_full = np.zeros((states,states))
    for ii in range(states):
        pi_full[ii] = pi[lookup[:,ii]].prod()
        for jj in range(states):
            Gam_full[ii,jj] = Gam[lookup[:,ii],lookup[:,jj]].prod()
    return Gam_full, pi_full, lookup

def make_big_pi(pi_in,lookup,states_on=1):
    states_on = np.atleast_1d(states_on)

    pi_in = pi_in.ravel()
    components = len(pi_in)

    pi = np.zeros((lookup.shape[-1],components))
    pi[:,0] = pi_in[0]
    for k in range(1,components):
        idx_on = np.isin(lookup[k-1],states_on)
        pi[idx_on,k] = pi_in[k]

    pi /= pi.sum(-1)[:,None]
    return pi

class MM_Finite(Module):
    def __init__(self,output_dim=1,states=3,learn=False,components=2):
        super().__init__()

        self.states = states ** (components-1)
        self.learn = learn
        self.components = components

        self.mix = Mixture(components=components,learn=False)
        self.hmm = HMM(states=self.states,parameter_sampling=False)

        theta = [NormalWishart(hyper_sample=True,full_covariance=False,sigma_ev=2,transform_sample=False)]
        theta += [NormalWishart(hyper_sample=True,full_covariance=False,sigma_ev=.5,transform_sample=True) for k in range(components-1)] 
        self.theta = Plate(*theta)
        self.theta[0]._parameters['A'][:] = 0

        # Set HMM / Mix
        self.states_on = [1,2]
        ev = np.array([100,0,100,100])
        ev_p = ev/(ev+1)
        Gam = np.diag(ev_p)

        Gam = np.eye(states)
        Gam[0,1] = 1-ev_p[0]
        Gam[1,2] = 1-ev_p[1]
        Gam[2,3] = 1-ev_p[2]
        Gam[-1,0] = 1-ev_p[-1]
        pi = np.zeros(states)
        pi[0] = 1
        pi[1] = .5
        Gam /= Gam.sum(-1)[:,None]
        pi /= pi.sum()

        Gam_full, pi_full, self.lookup = make_big_hmm(Gam,pi,components)
            
        self.hmm.set_parameters(Gamma=Gam_full,pi=pi_full)

        self.mix._parameters['pi'] = np.ones(components)/components
        self.logpi = np.log(make_big_pi(self.mix.pi,lookup=self.lookup,states_on=self.states_on))

    def _sample_hmm(self,T):
        logl_hmm = self.logpi[:,self.mix.z].reshape(self.states,T,-1).sum(-1).T
        self.hmm.forward(logl_hmm)
    
    def _loglikelihood(self,data:'Data'):
        logl = np.zeros((len(data),self.components))
        for k in range(self.components):
            logl[:,k] = mvn.logpdf(data.output[:,0],self.theta[k].A,self.theta[k].Q)
        return logl

    def _sample_mix(self,data:'Data'):
        logl = self._loglikelihood(data=data)
        rho = logl + self.logpi[self.hmm.z[time]]
        rho -= logsumexp(rho,-1).reshape(-1,1)
        rho = np.exp(rho)
        rho /= rho.sum(-1)[:,None]
        for n in range(rho.shape[0]):
            self.mix._parameters['z'][n] = np.random.multinomial(1,rho[n]).argmax()

    def _sample_mix_pi(self):
        alpha = np.zeros(self.components)
        for k in range(self.components):
            if k == 0:
                any_of_these = np.nonzero(np.any(np.isin(model.lookup,self.states_on),0))[0]
                n_on = np.isin(self.hmm.z[data.time],any_of_these)
            else:
                n_on = np.isin(self.lookup[k-1,self.hmm.z[data.time]],self.states_on)
            alpha[k] = (self.mix.z[n_on]==k).sum() + self.mix.alpha0[k]
        self.mix._parameters['pi'] = dirichlet.rvs(alpha).ravel()
        self.logpi = np.log(make_big_pi(self.mix.pi,lookup=self.lookup,states_on=self.states_on))

    def _sample_parameters(self,data:'Data'):
        self.theta.forward(y=data.output,labels=self.mix.z)
        
    def forward(self,data:'Data'):
        N = len(data)
        if self.mix.z.shape[0] != N:
            mz_init = np.random.randint(0,self.components,N)*0
            self.mix._parameters['z'] = mz_init + 0

        self._sample_hmm(T=data.T)
        self._sample_mix(data=data)
        self._sample_mix_pi()
        if self.learn:
            self._sample_parameters(data=data)
            
#%%
sigma1 = 2
sigma2 = .1
mu1 = 0
mu2 = 0

N = 6
T = 150

np.random.seed(123)

y1 = np.random.normal(mu1,sigma1,(T,N))
y2 = np.random.normal(mu2-1,sigma2,(T,N))
y3 = np.random.normal(mu2+1,sigma2*.5,(T,N))
y = y2.copy()
y[T//2:] = y3[T//2:]
y[:20] = y1[:20]
y[-20:] = y1[-20:]
y[T//2-20:T//2+20] = y1[T//2-20:T//2+20]
y[:,-N//4:] = y1[:,-N//4:]
y = np.sort(y,-1)
y = y.ravel()
time = np.arange(T).repeat(N)
data = Data(y=y[:,None],time=time)
plt.plot(time,y,'.')
plt.xlim(0,T-1)

#%%
model = MM_Finite(components=4,states=4,learn=True)
sampler = Gibbs()

#%%
sampler.fit(data=data,model=model,samples=200)

# %%
chain = sampler.get_chain(burn_rate=.5)
z_hat = categorical2multinomial(chain['mix.z']).mean(0).argmax(-1)
hmm_z_hat = categorical2multinomial(chain['hmm.z']).mean(0).argmax(-1)


# %%
fig, ax = plt.subplots(2,figsize=(4,3),sharex=True,gridspec_kw={'height_ratios': [1, 3]})
colors = get_colors()
ax[1].scatter(time,y,c=colors[z_hat],linewidths=0,s=5,alpha=.8)
ax[1].set_xlim(0,T)
ax[0].imshow(colors[model.lookup[:,hmm_z_hat]])
ax[0].set_yticks(np.arange(model.components-1),np.arange(model.components-1)+1)
plt.tight_layout()

fig,ax = plt.subplots(2,figsize=(4,3),sharex=False)
ax[0].imshow(chain['mix.z'])
ax[1].imshow(chain['hmm.z'])
plt.tight_layout()

fig,ax = plt.subplots(2,figsize=(4,3),sharex=True)
for i in range(model.components):
    ax[0].plot(chain['mix.pi'][:,i],color=colors[i])
for i in range(model.components):
    ax[1].plot(chain['theta.{}.Q'.format(i)][:,0,0]**.5,color=colors[i])
ax[1].set_xlim(0,chain['mix.pi'].shape[0])
plt.tight_layout()

# %%
