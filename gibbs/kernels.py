"""
Gaussian process kernels.

Julian Neri
McGill University
August 19, 2023

"""

import numpy as np
from scipy.special import gamma, kv
from typing import List


class Kernel(object):
    """
    Base class for Kernel.
    """
    def __init__(self,derive_dim:int=None) -> None:
        self.derive_dim = derive_dim

    def __call__(self,x,y):
        return self.forward(x,y)
    
    def forward(self,x,y):
        if y.ndim != 2:
            raise ValueError("y must be 2d")
        if x.ndim != 2:
            raise ValueError("x must be 2d")
        return 0
    
    def derivative_x(self,x,y,dim:int=0):
        return 0
    
    def derivative_y(self,x,y,dim:int=0):
        return 0
    
    def derivative_xy(self,x,y,dim:int=0):
        return 0

    def kernel_function(self,x,y):
        return 0
    
    def forward(self,x,y):
        if y.ndim != 2:
            raise ValueError("y must be 2d")
        if x.ndim != 2:
            raise ValueError("x must be 2d")
        K = self.kernel_function(x,y)
        if self.derive_dim is not None:
            K_dx = self.derivative_x(x,y,dim=self.derive_dim)
            K_dy = self.derivative_y(x,y,dim=self.derive_dim)
            K_dxy = self.derivative_xy(x,y,dim=self.derive_dim)
            K = np.vstack((np.hstack((K, K_dy)), np.hstack((K_dx, K_dxy)))) 
        return K
    
class ConstantKernel(Kernel):
    def __init__(self, sigma=1, derive_dim: int = None) -> None:
        self.sigma = sigma
        super().__init__(derive_dim)
    
    def derivative_x(self,x,y,dim:int=0):
        return np.zeros((x.shape[0],y.shape[0]))
    
    def derivative_y(self,x,y,dim:int=0):
        return np.zeros((x.shape[0],y.shape[0]))
    
    def derivative_xy(self,x,y,dim:int=0):
        return np.zeros((x.shape[0],y.shape[0]))

    def kernel_function(self,x,y):
        return self.sigma**2 * np.ones((x.shape[0],y.shape[0]))

    
class LinearKernel(Kernel):
    def __init__(self,scale=.2,sigma=1,derive_dim:int=None) -> None:
        self.sigma = sigma
        self.derive_dim = derive_dim
        self.scale = np.atleast_1d(scale)

    def derivative_x(self,x,y,dim=0):
        _x,_y = x[:,[dim]], y[:,[dim]].T
        dx = self.scale[dim] * np.ones_like(_x) * _y
        return self.sigma**2 * dx
    
    def derivative_y(self,x,y,dim=0):
        _x,_y = x[:,[dim]], y[:,[dim]].T
        dy = self.scale[dim] * _x * np.ones_like(_y)
        return self.sigma**2 * dy
    
    def derivative_xy(self,x,y,dim=0):
        _x,_y = x[:,[dim]], y[:,[dim]].T
        dxy = self.scale[dim] * np.ones_like(_x) * np.ones_like(_y)
        return self.sigma**2 * dxy 

    def kernel_function(self,x,y):
        _x = (x.T)[:,:,None]
        _y = (y.T)[:,None,:]
        scale = self.scale.reshape(-1,1,1)
        K = scale * _x*_y
        K = K.sum(0)
        return self.sigma**2 * K

class RBF(Kernel):
    def __init__(self,scale=.2,sigma=1,derive_dim:int=None) -> None:
        self.sigma = sigma
        self.derive_dim = derive_dim
        self.scale = np.atleast_1d(scale)

    def derivative_x(self,x,y,dim=0):
        _x,_y = x[:,[dim]], y[:,[dim]].T
        s = 2 * self.scale[dim] ** 2
        dx = -2*(_x-_y)/s 
        return dx * self.kernel_function(x,y)
    
    def derivative_y(self,x,y,dim=0):
        _x,_y = x[:,[dim]], y[:,[dim]].T
        s = 2 * self.scale[dim] ** 2
        dy = 2*(_x-_y)/s 
        return dy * self.kernel_function(x,y)
    
    def derivative_xy(self,x,y,dim=0):
        _x,_y = x[:,[dim]], y[:,[dim]].T
        s = 2 * self.scale[dim] ** 2
        dxy = 2*(s - 2*(_x-_y)**2) / s**2 
        return dxy * self.kernel_function(x,y)

    def kernel_function(self,x,y):
        _x = (x.T)[:,:,None]
        _y = (y.T)[:,None,:]
        scale = self.scale.reshape(-1,1,1)
        inside = -.5*((_x - _y)/scale)**2
        K = np.exp(inside.sum(0))
        return self.sigma**2 * K

class Matern(Kernel):
    def __init__(self,p=1,scale=.2,sigma=1,derive_dim:int=None) -> None:
        self.nu = p + 1/2
        self.scale = np.atleast_1d(scale)
        self.sigma = sigma
        self.derive_dim = derive_dim

    def derivative_x(self,x,y,dim=0):
        _x,_y = x[:,[dim]], y[:,[dim]].T
        l = self.scale[dim]
        s = np.sqrt(3)
        d = np.abs(_x - _y)
        dx = s ** 2 * (_y - _x)  / (l*(l+s*d))
        return dx * self.kernel_function(x,y)
    
    def derivative_y(self,x,y,dim=0):
        _x,_y = x[:,[dim]], y[:,[dim]].T
        l = self.scale[dim]
        s = np.sqrt(3)
        d = np.abs(_x - _y)
        dy = s ** 2 * (_x - _y)  / (l*(l+s*d))
        return dy * self.kernel_function(x,y)
    
    def derivative_xy(self,x,y,dim=0):
        _x,_y = x[:,[dim]], y[:,[dim]].T
        l = self.scale[dim]
        s = np.sqrt(3)
        d = np.abs(_x - _y)
        dxy = (s/l) ** 2 * (-1 + 2*l  / (l+s*d))
        return dxy * self.kernel_function(x,y)

    def kernel_function(self,x,y):
        _x = (x.T)[:,:,None]
        _y = (y.T)[:,None,:]
        l = self.scale.reshape(-1,1,1)
        d = np.abs(_x - _y)

        K = (1+np.sqrt(3)*d/l)*np.exp(-np.sqrt(3)*d/l)
        K = K.prod(0)
        return self.sigma**2 * K
    
class Composition(Kernel):
    def __init__(self,list:List[Kernel]) -> None:
        self.list = list

    def forward(self,x,y):
        K = 0
        for l in self.list:
            K += l.forward(x,y)
        return K
    

class ArcCos(Kernel):
    def __init__(self,sigma_w=[4,3],sigma_b=[4,2],order:int=1) -> None:
        sigma_w = np.asarray(sigma_w)
        sigma_b = np.asarray(sigma_b)
        self.sigma2_w = sigma_w ** 2
        self.sigma2_b = sigma_b ** 2
        self.layers = len(sigma_w) - 1
        self.order = order

    def J(self,theta):
        if self.order == 0:
            J = np.pi - theta
        elif self.order == 1:
            J = np.sin(theta) + (np.pi - theta) * np.cos(theta)
        else:
            J = 3*np.sin(theta)*np.cos(theta) + (np.pi - theta) *(1 + 2*np.cos(theta)**2)
        return J

    def k0(self,x,y,l=0):
        return self.sigma2_b[l] + self.sigma2_w[l] * x * y
    
    def kl(self,kx,ky,kxy,l:int):
        den = np.sqrt(kx * ky)

        inside = kxy  / den
        inside[inside>1] = 1
        theta = np.arccos(inside)

        k = self.sigma2_b[l] + self.sigma2_w[l]/(2*np.pi)* den * self.J(theta)
        return k
    
    def forward(self,x,y):
        x = np.atleast_1d(x).ravel()[:,None]
        y = np.atleast_1d(y).ravel()[None,:]
        
        kxy_prev = self.k0(x,y)
        kx_prev = self.k0(x,x)
        ky_prev = self.k0(y,y)
        
        for l in range(self.layers):
            kxy_l = self.kl(kx=kx_prev,ky=ky_prev,kxy=kxy_prev,l=l+1)

            if l < (self.layers - 1):
                kx_l = self.kl(kx=kx_prev,ky=kx_prev,kxy=kx_prev,l=l+1)
                ky_l = self.kl(kx=ky_prev,ky=ky_prev,kxy=ky_prev,l=l+1)

                kxy_prev = kxy_l.copy()
                kx_prev = kx_l.copy()
                ky_prev = ky_l.copy()

        return kxy_l

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use('gibbs.mplstyles.latex')

    N = 101
    M = 4

    x = np.linspace(-1,1,N)
    mu = np.zeros(N)

    kernel_rbf = RBF(scale=.2,sigma=1)
    K_rbf = kernel_rbf(x.reshape(-1,1),x.reshape(-1,1))
    C_rbf = np.eye(K_rbf.shape[0])*1e-8 + K_rbf
    y_rbf = np.random.multivariate_normal(mu,C_rbf,M).T

    kernel_matern = Matern(p=2,scale=.2)
    K_matern = kernel_matern(x.reshape(-1,1),x.reshape(-1,1))
    C_matern = np.eye(K_matern.shape[0])*1e-8 + K_matern
    y_matern = np.random.multivariate_normal(mu,C_matern,M).T

    fig,ax = plt.subplots(1,2,figsize=(5,2),sharey=True)
    ax[0].plot(x,y_rbf)
    ax[1].plot(x,y_matern)
    plt.tight_layout()
    

