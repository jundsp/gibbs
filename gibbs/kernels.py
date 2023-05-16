import numpy as np

class RBF(object):
    def __init__(self,sigma_cont=1,sigma_rbf=1,sigma_lin=1,scale=.2) -> None:
        self.sigma_cont = sigma_cont
        self.sigma_lin = sigma_lin
        self.sigma_rbf = sigma_rbf
        self.scale = scale

    def __call__(self,x,y):
        return self.forward(x,y)
    
    def forward(self,x,y):
        x = np.atleast_1d(x).ravel()[:,None]
        y = np.atleast_1d(y).ravel()[None,:]

        k_const = 1
        k_linear = x * y
        k_rbf =  np.exp(-.5*((x - y)/self.scale)**2)

        K = self.sigma_cont**2*k_const + self.sigma_lin**2*k_linear + self.sigma_rbf**2*k_rbf
        return K
    


class RBF_2d(object):
    def __init__(self,sigma_cont=1,sigma_rbf=1,sigma_lin=1,scale_rbf=[.2,.2],scale_lin=[1,1]) -> None:
        self.sigma_cont = sigma_cont
        self.sigma_lin = sigma_lin
        self.sigma_rbf = sigma_rbf
        self.scale_rbf = scale_rbf
        self.Sigma_lin = np.diag(scale_lin)

    def __call__(self,x,y):
        return self.forward(x,y)
    
    def forward(self,x,y):
        x= np.atleast_2d(x)
        y = np.atleast_2d(y)

        k_const = 1
        k_linear = x @ self.Sigma_lin @ y.T
        inside = 0
        for i in range(x.shape[-1]):
            inside += -.5*((x[:,[i]] - y[:,[i]].T)/self.scale_rbf[i])**2
        k_rbf =  np.exp(inside)

        K = self.sigma_cont**2*k_const + self.sigma_lin**2*k_linear + self.sigma_rbf**2*k_rbf
        return K



class ArcCos(object):
    def __init__(self,sigma_w=[4,3],sigma_b=[4,2],order:int=1) -> None:
        sigma_w = np.asarray(sigma_w)
        sigma_b = np.asarray(sigma_b)
        self.sigma2_w = sigma_w ** 2
        self.sigma2_b = sigma_b ** 2
        self.layers = len(sigma_w) - 1
        self.order = order

    def __call__(self, x, y):
        return self.forward(x=x,y=y)
    
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