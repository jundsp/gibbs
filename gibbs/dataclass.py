import numpy as np
import matplotlib.pyplot as  plt
import sines
import soundfile as sf

class Data(object):
    def __init__(self,y: np.ndarray,mask:np.ndarray=None) -> None:
        self.value = y
        self.mask = mask
        
    @property
    def value(self) -> np.ndarray:
        return self._value

    @value.setter
    def value(self,val: np.ndarray) -> None:
        if val.ndim != 3:
            raise ValueError("data values must be 3d: (time x obs x dim)")
        self._value = val
        self._T,self._N,self._M = val.shape

    @property
    def mask(self) -> np.ndarray:
        return self._mask
    @mask.setter
    def mask(self,val: np.ndarray) -> None:
        if val is None:
            val = np.ones(self.value.shape[:2]).astype(bool)
        if val.ndim != 2:
            raise ValueError("mask must be 2d")
        if val.shape[:2] != self.value.shape[:2]:
            raise ValueError("fist 2 dims of value and mask must match")
        
        val = np.atleast_2d(val).astype(bool)
        self._mask = val

        self._nonzero()

    def _nonzero(self):
        self._rnz, self._cnz = np.nonzero(self.mask)

    def flatten(self):
        return self.value[self._rnz,self._cnz]

    def time(self):
        return self._rnz

    @property
    def T(self) -> int:
        return self._T
    @property
    def N(self) -> int:
        return self._N
    @property
    def dim(self) -> int:
        return self._M

    def plot(self):
        plt.plot(self.time(),self.flatten(),'.')

    def __len__(self) -> int:
        return self.value.shape[0]

    def __repr__(self) -> str:
        output = self.__class__.__name__  + " \n"
        output += "values =  " + str(self.value) + " \n"
        output += "mask =  " + str(self.mask) + " \n"
        return output


if __name__ == "__main__":
    audio,sr = sf.read("/Users/julian/Documents/MATLAB/sounds/greasy.wav")
    sm = sines.Sines(step_size=2,confidence=.9,sr_down=16e3,resolutions=1,cent_threshold=10,window_size=20)
    features = sm(audio,sr)

    frame = features['frame']
    t = np.unique(frame)
    T = t.max()
    xx = features['frequency']
    yy = features['amplitude'][:,None]
    M = yy.shape[-1]

    from itertools import groupby
    group = groupby(frame)
    N = max(group, key=lambda k: len(list(k[1])))[0]
    y = np.zeros((T,N,M))
    x = np.zeros((T,N))
    mask = np.zeros((T,N)).astype(bool)
    for t in range(T):
        idx = frame==t
        n = idx.sum()
        x[t,:n] = xx[idx]
        y[t,:n] = yy[idx]
        mask[t,:n] = True

    data = Data(y=y,mask=mask)

    print(data.flatten())
    print(data.time())

    data.plot()
