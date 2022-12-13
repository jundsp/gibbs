import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multinomial, wishart
from scipy.stats import multivariate_normal as mvn
import scipy.linalg as la

def plot_cov_ellipse(pos,cov, nstd=2, ax=None, fill=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, fill=fill, **kwargs)

    ax.add_artist(ellip)
    return ellip

def get_median(stacked):
        return np.median(stacked,0).astype(stacked.dtype)

def get_mean(stacked):
    return np.mean(stacked,0)
    
def mvn_logpdf(y,mu,Sigma):
    # T x M or T X N X M
    y_eps = y-mu
    iSigma = la.inv(Sigma)[None,:,:]
    quad = (y_eps[:,None,:] @ iSigma @ y_eps[:,:,None]).ravel()
    return -0.5*(np.linalg.slogdet(2*np.pi*Sigma)[-1] + quad)
    
    
def gmm_generate(n=100,output_dim=2,n_components=3):
    mu = mvn.rvs(np.zeros(output_dim),np.eye(output_dim)*5,n_components
    )
    nu = output_dim+.1
    W = np.eye(output_dim)*5.0 / nu

    Lambda = wishart.rvs(nu,W,n_components)
    Sigma = [la.inv(Lambda[i]) for i in range(n_components)]

    z = np.random.randint(0,n_components,n).astype(int)
    x = np.zeros((n,output_dim))
    for i in range(n):
        x[i] = np.random.multivariate_normal(mu[z[i]],Sigma[z[i]])
    return x, z

def hmm_generate(n=100,output_dim=2,n_components=3,expected_duration=10):
    A_kk = expected_duration / (expected_duration+1)
    A_jk = 1.0
    if n_components > 1:
        A_jk = (1-A_kk) / (n_components-1)
    Gamma = np.ones((n_components,n_components)) * A_jk
    np.fill_diagonal(Gamma,A_kk)
    Gamma /= Gamma.sum(-1).reshape(-1,1)
    pi = np.ones(n_components) / n_components

    mu = np.random.normal(0,2,(100,output_dim))
    sigmas = np.random.gamma(shape=3,scale=.1,size=n_components)
    sigmas[0] = 2
    Sigma = np.stack([np.eye(output_dim)*s for s in sigmas],0)

    z = np.zeros(n).astype(int)
    x = np.zeros((n,output_dim))
    predict = pi.copy()
    for i in range(n):
        z[i] = multinomial.rvs(1,predict).argmax()
        x[i] = np.random.multivariate_normal(mu[z[i]],Sigma[z[i]])
        predict = Gamma[z[i]]
    return x, z

def lds_generate(T=100):
    t = np.arange(T)
    y = np.exp(1j*2*np.pi*4*t/100)
    y = np.stack([y.real,y.imag],-1)
    y += np.random.normal(0,.05,y.shape)
    return y

def slds_generate(T=100):
    T1 = T//2
    t = np.arange(T1)
    y1 = np.exp(1j*2*np.pi*4*t/100)*np.exp(-2*t/100)
    y2 = np.exp(1j*2*np.pi*1*t/100-1j*np.pi)*np.exp(-t/100)
    y3 = np.exp(1j*2*np.pi*.25*t/100-1j*np.pi)*np.exp(-t/100)
    y = np.concatenate([y1,y2],0)
    y = np.stack([y.real,y.imag],-1)
    y += np.random.normal(0,.01,y.shape)
    return y

def get_colors():
    k = np.arange(8)
    cl = ['Dark2', 'Set1', 'Set2']
    colors = np.concatenate([plt.get_cmap(s)(k) for s in cl],0)
    return colors

def get_scatter_kwds():
    kwds = dict(alpha=.5,s=20,edgecolor='none')
    return kwds

def scattercat(y,z,figsize=(4,3)):
    plt.figure(figsize=figsize)
    colors = get_colors()
    plt.scatter(y[:,0],y[:,1],c=colors[z],**get_scatter_kwds())
    plt.xlabel('$y_1$')
    plt.ylabel('$y_2$')
    plt.tight_layout()

def makesymmetric(A):
    return .5*(A + A.T)