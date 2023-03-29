#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma, norm, dirichlet, multivariate_normal as mvn
from itertools import product
import os
from pathlib import Path
from gibbs import Gibbs, get_colors, Data, Mixture, Module, HMM, logsumexp, categorical2multinomial, NormalWishart, Plate, classification_accuracy, relabel

import argparse

parser = argparse.ArgumentParser(description='GHMM Finite example')
parser.add_argument('--output-directory', type=str, default='.', metavar='fname', help='Folder to save output data')
parser.add_argument('--samples', type=int, default=200, metavar='fname', help='Number of samples')
args = parser.parse_args()
# plt.style.use('sines-latex')
plt.style.use('gibbs.mplstyles.latex')

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

def make_hmm_params(expected_durations):
    states = len(expected_durations)
    ev_p = expected_durations/(expected_durations+1)

    Gam = np.diag(ev_p)
    for i in range(states-1):
        Gam[i,i+1] = 1 - ev_p[i]
    Gam[-1,0] = 1-ev_p[-1]
    Gam /= Gam.sum(-1)[:,None]

    pi = np.zeros(states)
    pi[0] = 1
    pi[1] = .5
    pi /= pi.sum()
    return Gam, pi

def test_data(T=100,N=10):
    sigma = np.array([2,.1,.1])
    sigma[1:] = np.random.uniform(1e-3,.2,2)
    mu = np.zeros(3)
    mu[1] = np.random.uniform(-2,-.5)
    mu[2] = np.random.uniform(.5,2)

    y = np.zeros((T,N))
    labels = np.zeros((T,N)).astype(int)
    

    starting = np.random.randint(0,T//2,2)
    ending = starting + np.random.randint(T//4,T-T//4,2)

    labels[starting[0]:ending[0],:N//3] = 1
    labels[starting[1]:ending[1],N//3:N//3*2] = 2

    for i in range(3):
        _y = np.random.normal(mu[i],sigma[i],(T,N))
        y[labels==i] = _y[labels==i]

    srt = np.argsort(y,-1)
    y = np.stack([y[i,srt[i]] for i in range(T)],0)
    labels = np.stack([labels[i,srt[i]] for i in range(T)],0)

    labels = labels.ravel()
    y = y.ravel()
    time = np.arange(T).repeat(N)

    data = Data(y=y[:,None],time=time)  
    return data, labels


def evaluate(path,trials=10,samples=100,burn_rate=.75,T=100,N=10, components=6, states=3, hyper_sample=False, save_plots=True):
    path = Path(path)
    os.makedirs(path,exist_ok=True)
    filename = path / "evaluation.txt"
    with open(filename, 'w') as f:
        f.write('')

    accuracy = np.zeros(trials)
    for trial in range(trials):
        data, labels = test_data(T=T,N=N)
        model = MM_Finite(components=components,states=states,learn=True,hyper_sample=hyper_sample)
        sampler = Gibbs()

        sampler.fit(data=data,model=model,samples=samples)

        chain = sampler.get_chain(burn_rate=burn_rate)
        tau = relabel(probs=chain['mix.rho'],verbose=True,iters=10)

        rho = chain['mix.rho']
        rho = np.take_along_axis(rho,tau[:,None,:],-1)
        z_hat = rho.mean(0).argmax(-1)

        accuracy[trial] = classification_accuracy(labels, z_hat,M=3,K=model.components)
        with open(filename, 'a') as the_file:
            the_file.write('{}\t{}\t{:4.3f}\n'.format(trial,samples,accuracy[trial]))

        if save_plots:
            hmm_z_hat = categorical2multinomial(chain['hmm.z']).mean(0).argmax(-1)

            colors = get_colors()
            fig, ax = plt.subplots(2,figsize=(4,3),sharex=True,gridspec_kw={'height_ratios': [1, 3]})
            ax[1].scatter(data.time,data.output[:,0],c=colors[z_hat],linewidths=0,s=5,alpha=.8)
            ax[1].set_xlim(0,data.T-1)
            ax[0].imshow(colors[model.lookup[:,hmm_z_hat]])
            ax[0].set_yticks(np.arange(model.components-1),np.arange(model.components-1)+1)
            plt.tight_layout()
            plt.savefig(path/ "eval_data.png")

            fig,ax = plt.subplots(2,figsize=(4,3),sharex=False)
            ax[0].imshow(chain['mix.z'])
            ax[1].imshow(chain['hmm.z'])
            plt.tight_layout()
            plt.savefig(path/ "eval_chain.png")

            fig,ax = plt.subplots(3,figsize=(4,3),sharex=True)
            for i in range(model.components):
                ax[0].plot(chain['mix.pi'][:,i],color=colors[i])
                ax[1].plot(chain['theta.{}.Q'.format(i)][:,0,0]**.5,color=colors[i])
                ax[2].plot(chain['theta.{}.A'.format(i)][:,0,0],color=colors[i])
            ax[-1].set_xlim(0,chain['mix.pi'].shape[0])
            plt.tight_layout()
            plt.savefig(path/ "eval_params.png")

    return accuracy

class MM_Finite(Module):
    def __init__(self,output_dim=1,states=3,components=2,learn=True,hyper_sample=True):
        super().__init__()

        self.states_on = np.arange(states-2)+1
        self.states = states ** (components-1)
        self.learn = learn
        self.components = components

        self.mix = Mixture(components=components,learn=False)
        self.hmm = HMM(states=self.states,parameter_sampling=False)

        theta = [NormalWishart(hyper_sample=hyper_sample,full_covariance=False,sigma_ev=2,transform_sample=False)]
        theta += [NormalWishart(hyper_sample=hyper_sample,full_covariance=False,sigma_ev=.5,transform_sample=True) for k in range(components-1)] 
        self.theta = Plate(*theta)
        for k, theta in enumerate(self.theta):
            theta._parameters['A'][:] = 0

        # Set HMM / Mix
        expected_durations = np.ones(states)*100
        if states > 3:
            expected_durations[1:(states-3)+1] = 0
        Gam,pi = make_hmm_params(expected_durations=expected_durations)
        self._Gamma_template = Gam
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
        rho = logl + self.logpi[self.hmm.z[data.time]]
        rho -= logsumexp(rho,-1).reshape(-1,1)
        rho = np.exp(rho)
        rho /= rho.sum(-1)[:,None]
        self.mix._parameters['rho'] = rho.copy()
        for n in range(rho.shape[0]):
            self.mix._parameters['z'][n] = np.random.multinomial(1,rho[n]).argmax()

    def _sample_mix_pi(self,data:'Data'):
        alpha = np.zeros(self.components)
        for k in range(self.components):
            if k == 0:
                any_of_these = np.nonzero(np.any(np.isin(self.lookup,self.states_on),0))[0]
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
            self.mix._parameters['z'] = np.random.randint(0,self.components,N)

        self._sample_hmm(T=data.T)
        self._sample_mix(data=data)
        self._sample_mix_pi(data=data)
        if self.learn:
            self._sample_parameters(data=data)


# #%%
# np.random.seed(123)
# data, labels = test_data(T=100,N=10)

# colors = get_colors()
# plt.figure(figsize=(4,2))
# plt.scatter(data.time,data.output[:,0],c=colors[labels],linewidths=0,s=5,alpha=.8)
# plt.xlim(0,data.T-1)
# plt.title('Target')
# plt.tight_layout()

# # %%
# model = MM_Finite(components=5,states=3,learn=True,hyper_sample=False)
# sampler = Gibbs()

# np.random.seed(123)
# #%%
# #  Converges after 1000 samples. Hyper_sample = True converges faster, because it explores the space (mu, sigma) better than with alpha fixed. 
# sampler.fit(data=data,model=model,samples=args.samples)

# # %%
# chain = sampler.get_chain(burn_rate=0)
# z_hat = categorical2multinomial(chain['mix.z']).mean(0).argmax(-1)
# hmm_z_hat = categorical2multinomial(chain['hmm.z']).mean(0).argmax(-1)

# fig, ax = plt.subplots(2,figsize=(4,3),sharex=True,gridspec_kw={'height_ratios': [1, 3]})

# ax[1].scatter(data.time,data.output[:,0],c=colors[z_hat],linewidths=0,s=5,alpha=.8)
# ax[1].set_xlim(0,data.T-1)
# ax[0].imshow(colors[model.lookup[:,hmm_z_hat]])
# ax[0].set_yticks(np.arange(model.components-1),np.arange(model.components-1)+1)
# plt.tight_layout()

# fig,ax = plt.subplots(2,figsize=(4,3),sharex=False)
# ax[0].imshow(chain['mix.z'])
# ax[1].imshow(chain['hmm.z'])
# plt.tight_layout()

# fig,ax = plt.subplots(2,2,figsize=(4,3),sharex=True)
# ax = ax.ravel()
# for i in range(model.components):
#     ax[0].plot(chain['mix.pi'][:,i],color=colors[i])
#     ax[1].plot(chain['theta.{}.Q'.format(i)][:,0,0]**.5,color=colors[i])
#     ax[2].plot(chain['theta.{}.A'.format(i)][:,0,0],color=colors[i])
#     ax[3].plot(chain['theta.{}.alpha'.format(i)],color=colors[i])
# ax[-1].set_xlim(0,chain['mix.pi'].shape[0]-1)
# plt.tight_layout()

# outpath = args.output_directory + "/results"
# os.makedirs(outpath,exist_ok=True)
# plt.savefig(outpath + '/ghmm_params_alpha_sampled_m0.png')


# #%%
# accuracy = classification_accuracy(labels, z_hat,M=3,K=model.components)
# print(r"Accuracy = {:3.1f}%".format(100*accuracy))


#%%
np.random.seed(123)
accuracy = evaluate(trials=1,samples=500,burn_rate=0,path=args.output_directory + '/results/ghmm/')

print(accuracy)
# plt.plot(accuracy)
# %%
