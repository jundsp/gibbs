import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multinomial

def plot_cov_ellipse(pos,cov, nstd=2, ax=None, **kwargs):
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
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def get_median(stacked):
        return np.median(stacked,0).astype(stacked.dtype)

def get_mean(stacked):
    return np.mean(stacked,0)
    

def gmm_generate(n=100,output_dim=2,n_components=3):
    mu = np.random.normal(0,2,(100,output_dim))
    sigmas = np.random.gamma(shape=3,scale=.1,size=n_components)
    sigmas[0] = 2
    Sigma = np.stack([np.eye(output_dim)*s for s in sigmas],0)

    z = np.random.randint(0,n_components,n).astype(int)
    x = np.zeros((n,output_dim))
    for i in range(n):
        x[i] = np.random.multivariate_normal(mu[z[i]],Sigma[z[i]])
    return x, z

def hmm_generate(n=100,output_dim=2,n_components=3):
    Gamma = np.eye(n_components) + 1e-1
    # Gamma *= 0
    # Gamma[:-1,1:] = np.eye(n_components-1)
    # Gamma[-1,0] = 1
    Gamma /= Gamma.sum(-1)[:,None]
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
