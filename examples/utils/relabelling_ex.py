import numpy as np
from gibbs.utils import relabel

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use('sines-latex')

    T = 200
    K = 5
    N = 800

    p = np.random.random((N,K))*1e-3
    p = np.random.multinomial(1,np.ones(K)/K,N).astype(float) + p
    p /= p.sum(-1)[:,None]

    tau_tru = np.zeros((T,K)).astype(int)
    _tau = np.arange(K)
    for t in range(T):
        tau_tru[t] = _tau + 0
        if (t+1) % (T//4) == 0:
            _tau = np.random.permutation(K)

    P = np.stack([p]*T,0)
    for t in range(T):
        P[t] = P[t,:,tau_tru[t]].T

    P += np.random.random(P.shape)*1e-2
    P /= P.sum(-1)[:,:,None]

    tau = relabel(P,verbose=True)

    fig,ax = plt.subplots(2)
    ax[0].imshow(tau_tru.T)
    ax[1].imshow(tau.T)

