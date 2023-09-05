import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multinomial, wishart
from scipy.stats import multivariate_normal as mvn, multivariate_t as mvt, wishart
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln, logsumexp

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
    
def gmm_generate(n=100,output_dim=2,n_components=3,sigma:float=1):
    mu = mvn.rvs(np.zeros(output_dim),np.eye(output_dim)*5,n_components)
    nu = output_dim+.5
    W = np.eye(output_dim)/sigma**2 / nu

    Lambda = wishart.rvs(nu,W,n_components)
    if output_dim == 1:
        mu = mu.reshape(-1,1)
        Lambda = Lambda.reshape(-1,1,1)
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
    kwds = dict(alpha=.5,s=15,edgecolor='none',linewidth=0)
    return kwds

def scattercat(y,z,figsize=(4,3),colors=None):
    plt.figure(figsize=figsize)
    if colors is None:
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

def relabel(probs,iters=10,verbose=False):
    """
    Relabeling algorithm to deal with label swithing in mixture models.

    Ref:
    M. Stephens, “Dealing with label switching in mixture models,” Journal of the Royal Statistical Society. Series B (Methodological), vol. 62, no. 4, pp. 795-809, 2000.

    Julian Neri
    August 2023
    """
    T,N,K = probs.shape

    tau = np.stack([np.arange(K)]*T,0)
    best_cost = np.inf

    q = np.zeros((N,K))
    for iter in range(iters):
        for i in range(N):
            for k in range(K):
                q[i,k] = probs[np.arange(T),i,tau[:,k]].mean(0) 

        logq = np.log(q+1e-9)

        total_cost = 0
        for t in range(T):
            p = probs[t][:,None,:]
            cost_matrix = np.sum(p*(np.log(p+1e-9) - logq[:,:,None]),0)
            r,c = linear_sum_assignment(cost_matrix)
            tau[t] = c
            total_cost += cost_matrix[r,c].sum()


        if verbose:
            print("Relabelling ==> KLD = {:4.2f}".format(total_cost))

        if total_cost < best_cost:
            best_cost = total_cost
        else:
            if verbose:
                print("Relabelling ==> Converged.")
            break

    return tau


def log_normalize(alpha):
    c = logsumexp(alpha)
    if np.isinf(c):
        alpha = np.zeros_like(alpha)
        c = logsumexp(alpha)
    alpha -= c
    return alpha, c


def gamma_moments2params(mu,var):
    a = mu**2 / var
    b = mu / var
    return a, b 

def gamma_params2moments(a,b):
    mu = a / b
    var = a / b ** 2
    return mu, var 

def invgamma_moments2params(mu,var):
    a = (mu**2 + 2*var) / var
    b = (mu*(mu**2 + var)) / var
    return a, b 

def invgamma_params2moments(a,b):
    mu = b / (a-1)
    var = b ** 2 / ((a-1)**2 * (a-2))
    return mu, var

def invgamma_b(mu,a=2):
    b = mu * (a-1)
    if b <= 0:
        b = mu * a
    return b

def invwishart_moments2params(mu,var):
    mu = np.atleast_1d(mu)
    p = len(mu)
    nu = (3 + p + 2*mu**2 / var).mean()
    Psi = 2*(mu**2 + mu*var) / var
    return nu, np.diag(Psi)

def invwishart_moments2params(mu,var):
    a,b = invgamma_moments2params(mu,var)
    nu = 2*np.min(a)
    psi = 2*b
    return nu, np.diag(psi)


def stdev_to_moments(stdev,variance_factor=.5):
    ev = stdev ** 2
    var = (variance_factor*stdev)**2
    return ev, var


class DirichletProcess(object):
    '''
    Dirichlet process model, for generation.

    Author: Julian Neri, 2022
    '''
    def __init__(self,alpha=1):
        self.alpha = alpha
        self.reset()
        
    def reset(self):
        self.Nk = []
        self.N = 0
        self.K = 0

    def sample(self):
        if self.K == 0:
            self.Nk = [1]
            self.K = 1
            z = 0
        else:
            pk = np.zeros(self.K+1)
            Z = self.N+self.alpha-1
            for k in range(self.K):
                pk[k] = self.Nk[k] / Z
            pk[-1] = self.alpha / Z
            pk /= pk.sum()
            z = np.random.multinomial(1,pk).argmax(-1)
            if z > self.K - 1:
                self.Nk.append(1)
                self.K += 1
            else:
                self.Nk[z] += 1
        self.N += 1
        return z


def mvt_logpdf(y:np.ndarray,loc:np.ndarray,shape:np.ndarray,df:int):
    """
    Multivariate-t distribution log pdf
    """
    if y.ndim != loc.ndim:
        raise ValueError("number of dims in y and params must match")
    if y.shape[0] != shape.shape[0]:
        raise ValueError("number of obs in y and shape must match")
    if y.shape[-1] != shape.shape[-1]:
        raise ValueError("dim of y must match shape")
    
    if y.ndim == 1:
        y = y[None,...]
        loc = loc[None,...]
        shape = shape[None,...]
    if y.ndim == 2:
        if shape.ndim == 2:
            shape = shape[None,...]
    # T x M or T X N X M
    Sigma = shape.copy()
    p = y.shape[-1]
    dp2 = (df + p) / 2
    y_eps = y-loc
    iSigma = np.linalg.inv(Sigma)
    quad = (y_eps[:,None,:] @ iSigma @ y_eps[:,:,None]).ravel()
    term1 = -dp2 * np.log(1 + 1/df * quad)
    term2 = gammaln(dp2)
    term3 = gammaln(df/2) + (p/2)*np.log(df) + p/2*np.log(np.pi) + 1/2*np.linalg.slogdet(Sigma)[-1]
    return (term1 + term2 - term3).ravel()

if __name__ == "__main__":

    N = 4
    y = np.random.randn(N,2)
    mu = np.random.randn(N,2)*0
    Sigma = wishart.rvs(df=2,scale=np.eye(2),size=N)
    # Sigma = np.stack([np.eye(2)]*N,0)
    nu = np.ones(1)*3.4

    p1 = np.zeros(N)
    for n in range(y.shape[0]):
        p1[n] = mvt.logpdf(y[n],loc=mu[n],shape=Sigma[n],df=nu)
    p2 = mvt_logpdf(y=y,loc=mu,shape=Sigma,df=nu)

    
    print(p1-p2)

