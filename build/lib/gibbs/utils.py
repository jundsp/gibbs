import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multinomial, wishart
from scipy.stats import multivariate_normal as mvn
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment

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

def lds_test_data(T=100):
    t = np.arange(T)
    y = np.exp(1j*2*np.pi*4*t/100)
    y = np.stack([y.real,y.imag],-1)
    y += np.random.normal(0,.05,y.shape)
    return y

def slds_test_data(T=100):
    T1 = T//3
    t = np.arange(T1)
    y1 = np.exp(1j*2*np.pi*4*t/100)*np.exp(-0*t/100)
    y2 = np.exp(1j*2*np.pi*1*t/100-1j*np.pi)*np.exp(-0*t/100)
    y3 = np.exp(1j*2*np.pi*3*t/100-1j*np.pi)*np.exp(-0*t/100)
    y = np.concatenate([y1,y2,y3],0)
    y = np.stack([y.real,y.imag],-1)
    y += np.random.normal(0,.01,y.shape)
    return y

def get_colors():
    k = np.arange(8)
    cl = ['Dark2', 'Set1', 'Set2']*2
    colors = np.concatenate([plt.get_cmap(s)(k) for s in cl],0)
    return colors

def get_scatter_kwds():
    kwds = dict(alpha=.5,s=15,edgecolor='none')
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

def categorical2multinomial(z:np.ndarray,n_categories:int=None):
    z = np.atleast_1d(z).astype(int)
    z_shape = z.shape
    z = z.ravel()
    if n_categories is None:
        n_categories = z.max()+1
    n_points = z.shape[0]
    rho = np.zeros((n_points,n_categories))
    rho[np.arange(n_points),z] = 1
    rho = rho.reshape(*(z_shape),-1)
    return rho

def mvnrnd(mu,Sigma,n=1):
    T = la.cholesky(Sigma)
    if np.iscomplexobj(Sigma):
        # eps = (np.random.randn(n,T.shape[0]) + 1j*np.random.randn(n,T.shape[0]))
        eps = np.random.randn(n,T.shape[0])
        x = eps @ T + mu[None,:]
        # eps_r = np.random.randn(n,T.shape[0])
        # eps_i = np.random.randn(n,T.shape[0])

        # x = eps_r @ T.real + (eps_i @ T.imag) * 1j + mu[None,:]
    else:
        eps = np.random.randn(n,T.shape[0])
        x = eps @ T + mu[None,:]
    return x

def classification_accuracy(target,estimate,M,K):
    target = categorical2multinomial(z=target,n_categories=M)
    estimate = categorical2multinomial(z=estimate,n_categories=K)
    cost_mtx = np.zeros((M,K))
    for m in range(M):
        for k in range(K):
            cost_mtx[m,k] = np.abs(target[:,m] - estimate[:,k]).sum()

    r,c = linear_sum_assignment(cost_matrix=cost_mtx)
    estimate = estimate[:,c]

    error = np.any((target - estimate) > 0,-1).sum()
    accuracy = 1.0 - error / target.shape[0]
    return accuracy


if __name__ == "__main__":
    z = np.random.randint(0,3,(3,4))
    rho = categorical2multinomial(z)
    print(rho)
    
    import matplotlib.pyplot as plt 
    nfft = 128
    w = np.hanning(nfft)
    F = np.fft.fft(np.eye(nfft))
    F = F * w[:,None]

    M = nfft//2+1
    F = F[:,:M]
    # plt.plot(F[:,nfft//2].imag)
    

    y = np.cos(2*np.pi*np.arange(nfft)/nfft*10.5) * w * np.exp(-np.linspace(0,6,nfft))*5

    sigma2 = .1**2
    Lam0 = np.eye(M)*1
    Lam = (F.conj().T @ F + Lam0)/sigma2
    ell = (F.conj().T @ y)/sigma2

    Sigma = la.inv(Lam)
    mu = Sigma @ ell

    x = mvnrnd(mu,Sigma,10).T
    y_hat = 2*(F @ x).real

    plt.plot(y_hat.real,'r',alpha=.1)
    plt.plot(y.real,'k')

    # x = mvnrnd(np.zeros(2)+1,np.eye(2),n=1000)
    # plt.figure()
    # plt.plot(x[:,0],x[:,1],'.')

    fig,ax = plt.subplots(2)
    ax[0].plot(x.real,'k',alpha=.5)
    ax[1].plot(x.imag,'r',alpha=.5)