import numpy as np

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

    @property
    def T(self) -> int:
        return self._T
    @property
    def N(self) -> int:
        return self._N
    @property
    def dim(self) -> int:
        return self._M

    def __len__(self) -> int:
        return self.value.shape[0]

    def __repr__(self) -> str:
        output = self.__class__.__name__  + " \n"
        output += "values =  " + str(self.value) + " \n"
        output += "mask =  " + str(self.mask) + " \n"
        return output


if __name__ == "__main__":
    y = np.random.randn(4,3,2)
    data = Data(y)
    mask = y[:,:,0].astype(bool)
    data.mask = mask
    print(data)
