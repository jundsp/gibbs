#%%
from gibbs import Gibbs, Module, Data, la, mvn, gamma, Plate, Mixture,mvn_logpdf,get_colors,get_scatter_kwds, CovarianceMatrix, TimePlate, HMM
import numpy as np
import os
import matplotlib.pyplot as plt
import sines
import soundfile as sf
#%%

from sequential.lds import polynomial_covariance_matrix,polynomial_matrix,polynomial_initial_covariance,polynomial_output_matrix

class PredictorStateSpace(Module):
    def __init__(self,output_dim=1,order=2,env_order=1,sample_obs=True,sample_sys=False,state='on'):
        super(PredictorStateSpace,self).__init__()
        self._dimy = output_dim
        self._order = order
        state_dim = order*env_order*output_dim
        self._dimx = state_dim
        self.sample_sys = sample_sys
        self.sample_obs = sample_obs
        self.state = state
        self._env_order = env_order
        self.initialize()
    
    def initialize(self):
        self.obs = CovarianceMatrix(dim=self.output_dim)
        self.sys = CovarianceMatrix(dim=self.state_dim)

        self.I = np.eye(self.state_dim)

        _I = np.eye(self._env_order)

        A,Q,C,m0,P0 = self._get_params()

        self._A = np.kron(_I,A)
        self._Q = np.kron(_I,Q)
        self._C = np.kron(_I,C)
        self._m0 = np.repeat(m0,self._env_order)
        self._P0 = np.kron(_I,P0)

        self.obs._parameters['cov'] = np.eye(self.output_dim)*.1
        self.sys._parameters['cov'] = self._Q

    def _get_params(self):
        A = polynomial_matrix(order=self._order)
        Q = polynomial_covariance_matrix(order=self._order)*1e-4
        C = polynomial_output_matrix(order=self._order)
        P0 = polynomial_initial_covariance(order=self._order,var_init=5)
        m0 = np.zeros(self._order)
        if self.state == 'off':
            A = np.zeros(A.shape)
            Q = np.eye(A.shape[0])*1e-9
            C = np.zeros(C.shape)
            P0 = Q
        elif self.state == 'initial':
            A = np.zeros(A.shape)
            Q = polynomial_initial_covariance(order=self._order,var_init=1)
            C = polynomial_output_matrix(order=self._order)
            P0 = polynomial_initial_covariance(order=self._order)
        return A,Q,C,m0,P0
            

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
        return self._A
    @property
    def Q(self):
        return self.sys.cov
    @property
    def C(self):
        return self._C
    @property
    def R(self):
        return self.obs.cov
    @property
    def m0(self):
        return self._m0
    @property
    def P0(self):
        return self._P0

    def forward(self,_y,_x):
        if self.sample_obs:
            self.obs(y=_y,x=_x)


class LDS_Predict(Module):
    r'''
        Bayesian linear dynamical system.

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self, output_dim=1, order=2, env_order=5, states=1):
        super(LDS_Predict,self).__init__()
        self._order = order
        state_dim = order*env_order
        self._dimy = output_dim
        self._dimx = state_dim
        self._dimz = states
        self._env_order = env_order

        self._parameters["x"] = np.zeros((1,state_dim))
        self.initialize()

    def initialize(self):
        mlist = []
        kwds = dict(output_dim=self.output_dim,env_order=self._env_order,order=self._order)
        desc = ['off'] + ['initial'] + ['on']*(self.states-2)
        desc = ['on']*self.states
        for i in range(self.states):
            mlist.append(PredictorStateSpace(state=desc[i],**kwds))
        self.theta = TimePlate(*mlist)
        self.I = np.eye(self.state_dim)
        self.env = sines.Envelope(order=self._env_order,looseness=.66)

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
    def C(self,k,x):
        return np.kron(self.env(x),np.eye(self._order))[::self._order]
    def R(self,k):
        return self.theta[0].R
    def m0(self,k):
        return self.theta[k].m0
    def P0(self,k):
        return self.theta[k].P0

    def transition(self,x,z:int=0):
        return self.A(z) @ x
        
    def emission(self,x,input,z:int=0):
        return self.C(z,x=input) @ x
    
    def predict(self,mu,V,z:int=0):
        m = self.transition(mu,z=z)
        P = self.A(z) @ V @ self.A(z).T+ self.Q(z)
        return m, (P)

    def update(self,time,data:'Data',m,P,z:int=0):
        idx = data.time == time
        y = data.output[idx]
        input = data.input[idx][:,0]
        Lam = la.inv(P)
        ell = Lam @ m

        for k in range(y.shape[0]):
            _C  = self.C(z,x=input[k])
            CR = _C.T @ la.inv(self.R(z))
            CRC = CR @ _C
            ell += CR @ y[k]
            Lam += CRC
        V = la.inv(Lam)
        mu = V @ ell
        return mu, (V)

    def _forward(self,data:'Data',z):
        self.T = data.T
        mu = np.zeros((self.T,self.state_dim))
        V = np.zeros((self.T,self.state_dim,self.state_dim))
        m = self.m0(z[0]).copy()
        P = self.P0(z[0]).copy()
        for n in range(self.T):
            ''' update'''
            mu[n], V[n] = self.update(n,data,m,P,z=z[n])
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

            _mu = mu[t] + K_star @ (self.x[t+1] - m)
            _V = (self.I - K_star @ self.A(state)) @ V[t]
            self._parameters['x'][t] = mvn.rvs(_mu,_V)

    def sample_x(self,data:'Data',z):
        mu, V = self._forward(data,z)
        self._backward(mu,V,z)


    def filtered_output(self,data:'Data',z):
        y_hat = np.zeros(data.output.shape)
        for t in np.unique(data.time):
            idx = data.time == t
            state = z[t]
            _C = self.C(state,data.input[idx][:,0])
            y_hat[idx,0] = _C @ self.x[t]
        return y_hat

    def loglikelihood(self,data:'Data',z):
        y_hat = self.filtered_output(data=data,z=z)
        logl = mvn_logpdf(data.output,y_hat,self.R(0))
        return logl
    
    def forward(self,data:'Data',z=None):
        if z is None:
            z = np.zeros(data.T).astype(int)        
        if self.x.shape[0] != data.T:
            self._parameters['x'] = np.zeros((data.T,self.state_dim))
        self.sample_x(data,z)
        y_hat = self.filtered_output(data,z)
        self.theta[0](data.output,y_hat)

class PredictorHMM(HMM):
    def __init__(self, states=1, expected_duration=1, parameter_sampling=True):
        super(PredictorHMM,self).__init__(states, expected_duration, parameter_sampling)

    def initialize(self):
        Gamma = np.zeros((self.states,self.states))
        Gamma[0,0] = 0
        Gamma[:-1,1:] = np.eye(self.states-1)
        Gamma[-1,-1] = 1
        Gamma[-1,0] = 0

        Gamma /= Gamma.sum(-1).reshape(-1,1)
        
        pi = np.zeros(self.states) 
        pi[:2] = 1
        pi /= pi.sum()

        self.Gamma0 = Gamma.copy()
        self.pi0 = pi.copy()

        self._parameters["Gamma"] = Gamma.copy()
        self._parameters["pi"] = pi.copy()


class SLDS_Predict(Module):
    r'''

        Gibbs sampling. 

        Author: Julian Neri, 2022
    '''
    def __init__(self, output_dim=1, order=2, env_order=5, states=1):
        super(SLDS_Predict,self).__init__()
        self.hmm = PredictorHMM(states=states,parameter_sampling=False)
        self.lds = LDS_Predict(output_dim=output_dim,order=order,env_order=env_order,states=states)
        self.states = states

    def logjoint(self,data:'Data'):
        logl = np.zeros((data.T, self.hmm.states))
        for s in range(self.hmm.states):
            m = self.lds.m0(s)[None,:]
            P = self.lds.P0(s)
            logl[0,s] += mvn_logpdf(self.lds.x[[0]],m,P)
 
            m = self.lds.x[:-1] @ self.lds.A(s).T 
            P = self.lds.Q(s)
            logl[1:,s] += mvn_logpdf(self.lds.x[1:],m,P)

            y_hat = self.lds.filtered_output(data=data,z=s*np.ones(data.T).astype(int))
            Sigma = self.lds.R(s)
            logly = mvn_logpdf(data.output,y_hat,Sigma)
            for t in np.unique(data.time):
                idx = data.time == t
                logl[t,s] += logly[idx].sum()
        return logl

    def forward(self,data:'Data'):
        if self.hmm.z.shape[0] != data.T:
            self.hmm._parameters['z'] = np.random.randint(0,self.hmm.states,data.T)

        self.lds(data=data,z=self.hmm.z)
        logl = self.logjoint(data)
        self.hmm(logl)

class EMM_LDS(Module):
    r'''
        Finite Bayesian mixture of Envelopes.

        Author: Julian Neri, 2022
    '''
    def __init__(self,components=3,output_dim=1, order=2, env_order=5, states=1):
        super(EMM_LDS,self).__init__()
        self.order = order
        self.components = components

        self.lds1 = Plate(*[LDS_Predict(output_dim=1, order=order, env_order=env_order, states=states) for i in range(self.components)])
        self.lds2 = Plate(*[LDS_Predict(output_dim=1, order=order, env_order=env_order, states=states) for i in range(self.components)])
        self.mix = Mixture(components=self.components)

    def loglikelihood(self,data: 'Data'):
        loglike = np.zeros((len(data),self.components))
        z = np.zeros((data.T)).astype(int)
        data_temp1 = Data(y=data.output[:,0][:,None],x=data.input,time=data.time)
        data_temp2 = Data(y=data.output[:,1][:,None],x=data.input,time=data.time)
        for k in range(self.components):
            
            loglike[:,k] = self.lds1[k].loglikelihood(data_temp1,z=z) + self.lds2[k].loglikelihood(data_temp2,z=z)
        return loglike

    def logjoint(self,data: 'Data'):
        logl = self.loglikelihood(data)
        temp = logl[np.arange(temp.shape[0]),self.mix.z]
        return temp

    def forward(self,data: 'Data'):
        if self.mix.z.shape[0] != len(data):
            self.mix._parameters['z'] = np.random.randint(0,self.components,(len(data)))
        for i in range(self.components):
            idx = self.mix.z == i
            data_temp = Data(y=data.output[idx,0][:,None],x=data.input[idx],time=data.time[idx])
            data_temp._T = data.T
            self.lds1[i](data_temp)

            data_temp = Data(y=data.output[idx,1][:,None],x=data.input[idx],time=data.time[idx])
            data_temp._T = data.T
            self.lds2[i](data_temp)
        self.mix(self.loglikelihood(data))


#%%
# np.random.seed(1)
filename = "source_example"
audio,sr = sf.read("/Users/julian/Documents/MATLAB/sounds/{}.wav".format(filename))
# audio += np.random.normal(0,1e-1,audio.shape)

time = np.arange(100)*.02
# time = None
sm = sines.Sines(confidence=.75,resolutions=1)
features = sm.short_term(audio,sr,time=time)

#%%
y1 = np.log(np.array(features['amplitude']))[:,None]
y2 = np.array(features['logfslope'])[:,None]
y = np.concatenate([y1,y2],-1)
x = np.array(features['frequency'])[:,None]/sr*2
t = np.array(features['frame'])

idx = (np.abs(y[:,1])  < 3) & (y[:,0] > -7)
data = Data(y=y[idx],x=x[idx],time=t[idx])
data.plot()
#%%
model = EMM_LDS(components=5,order=2,env_order=2,output_dim=1)
sampler = Gibbs()

#%%
sampler.fit(data,model,50)

#%%
sampler.get_estimates('median',burn_rate=.9)

z_hat = sampler._estimates['mix.z']
#%%
fig,ax = plt.subplots(1,2,figsize=(8,5),subplot_kw=dict(projection="3d",proj_type="ortho"))

colors = get_colors()
M = 32
input = np.linspace(0,1,M)
X, T = np.meshgrid(input,np.arange(data.T))
for d in range(2):
    y_env = np.zeros((model.components,data.T,M))
    for k in range(model.components):
        x_hat = sampler._estimates['lds{}.{}.x'.format(d+1,k)]
        R_hat = sampler._estimates['lds{}.{}.theta.0.obs.cov'.format(d+1,k)]
        for t in range(data.T):
            C = model.lds1[0].C(0,input)
            y_env[k,t] = C @ x_hat[t]


    for k in np.unique(z_hat):
        ax[d].plot_surface(T, X, y_env[k],alpha=.1,linewidth=0,color=colors[k],antialiased=True)
        
    ax[d].scatter(data.time,data.input,data.output[:,d],c=colors[z_hat],s=10,linewidth=0,edgecolor='none')
    ax[d].set_zlim(data.output[:,d].min(),data.output[:,d].max())
    ax[d].set_ylabel('frequency')
    ax[d].set_xlabel('time frame')
ax[0].view_init(30, -100)
ax[1].view_init(30,-105)
ax[0].set_zlabel('log amplitude')
ax[1].set_zlabel('log frequency slope')
plt.tight_layout()
path_out = "imgs"
os.makedirs(path_out,exist_ok=True)
plt.savefig(os.path.join(path_out,"envmix_ex.pdf"))
#%%
plot_chain = True
if plot_chain:
    chain = sampler.get_chain(burn_rate=.9,flatten=False)
    fig,ax = plt.subplots(len(chain),figsize=(5,1.5*len(chain)))
    for ii,p in enumerate(chain):
        if ('.x' in p) | ('.z' in p):
            _x = chain[p]
            _x = np.swapaxes(_x,0,1)
            _x = _x.reshape(_x.shape[0],-1)
            ax[ii].plot(_x,'k',alpha=.05)
        else:
            _x = chain[p]
            _x = _x.reshape(_x.shape[0],-1)
            ax[ii].plot(_x,alpha=.5)
        ax[ii].set_title(p)
    plt.tight_layout()
# %%
