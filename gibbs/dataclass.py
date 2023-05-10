import numpy as np
import matplotlib.pyplot as  plt
from .utils import get_colors

class Data(object):
    def __init__(self,y: np.ndarray=None,time:np.ndarray=None,x:np.ndarray=None,group:np.ndarray=None) -> None:
        self.load(y=y,time=time,x=x,group=group)

    def load(self,y: np.ndarray=None,time:np.ndarray=None,x:np.ndarray=None,group:np.ndarray=None) -> None:
        self.output = y
        self.time = time
        self.input = x
        self.group = group
        self.mask()
        
    @property
    def output(self) -> np.ndarray:
        return self._output

    @output.setter
    def output(self,val: np.ndarray) -> None:
        if val is None:
            val = np.zeros((0,0))
        if val.ndim != 2:
            raise ValueError("data values must be 2d: (obs x dim)")
        self._output = val
        self._L,self._dimy = val.shape

    @property
    def time(self) -> np.ndarray:
        return self._time
    @time.setter
    def time(self,val: np.ndarray) -> None:
        if val is None:
            val = np.zeros(self.output.shape[0])
        if val.ndim != 1:
            raise ValueError("time must be 1d")
        if val.shape[0] != self.output.shape[0]:
            raise ValueError("fist 2 dims of value and mask must match")
        
        val = np.atleast_1d(val).astype(int)
        self._time = val

        tunique = np.unique(val)
        if len(tunique) > 0:
            self._T = tunique.max()+1
        else:
            self._T = 0

    @property
    def output_dim(self) -> int:
        return self.output.shape[-1]
    
    @property
    def input(self) -> np.ndarray:
        return self._input
    @input.setter
    def input(self,val: np.ndarray) -> None:
        if val is None:
            val = np.zeros((self.output.shape[0],1))
        if val.ndim != 2:
            raise ValueError("input must be 2d")
        if val.shape[0] != self.output.shape[0]:
            raise ValueError("fist dim of input and output must match")

        self._input = val


    @property
    def group(self) -> np.ndarray:
        return self._group
    @group.setter
    def group(self,val: np.ndarray) -> None:
        if val is None:
            val = np.arange(len(self))
        if val.ndim != 1:
            raise ValueError("group must be 1d")
        if val.shape[0] != self.output.shape[0]:
            raise ValueError("fist dim of group and output must match")
        self._group = val
        self._num_groups = len(np.unique(val))

    @property
    def N(self):
        return self._num_groups
    
    @property
    def delta(self):
        return self._mask
    
    @property
    def min(self):
        return self.output.min()
    @property
    def max(self):
        return self.output.max()

    def y(self,t):
        return self.output[self.time == t]

    def x(self,t):
        return self.input[self.time == t]

    def count(self,t):
        return (self.time == t).sum()

    def mask(self,indices: np.ndarray=None):
        self._mask = np.zeros(self._L).astype(bool)
        if indices is None:
            self._mask[:] = True
        else:
            self._mask[indices] = True

    def filter(self,indices: np.ndarray=None,same_T=True):
        filtered_data = Data(y=self.output[indices],x=self.input[indices],time=self.time[indices],group=self.group[indices])
        if same_T:
            filtered_data._T = self.T
        return filtered_data


    @property
    def T(self) -> int:
        return self._T
    @property
    def L(self) -> int:
        return self.__len__()
    @property
    def dim(self) -> int:
        return self._dimy

    def load_block(self,y,mask=None,x=None):
        if y.ndim != 3:
            assert ValueError("Block output must be 3d.")
        if mask is None:
            mask = np.ones(y.shape[:2])
        if x is None:
            x = np.zeros((y.shape[0],y.shape[1],1))
        if mask.shape[:2] != y.shape[:2]:
            assert ValueError("dims must match in mask and output")
        if x.shape[:2] != y.shape[:2]:
            assert ValueError("dims must match in input and output")

        rz,cz = np.nonzero(mask)
        self.output = y[rz,cz]
        self.input = x[rz,cz]
        self.time = rz


    def plot(self,step_size=1):
        if self.T < 2:
            if self.dim == 2:
                plt.scatter(self.output[:,0],self.output[:,1],c='k',s=15)
            elif self.dim == 1:
                plt.scatter(self.input, self.output,c='k')
        else:
            fig,ax = plt.subplots(1,self.dim,figsize=(6,2))
            ax = np.atleast_1d(ax)
            for d in range(self.dim):
                ax[d].scatter(self.time[self.delta]*step_size,self.output[self.delta,d],alpha=.5,c='k',s=15,edgecolor='none')
            plt.tight_layout()

    def __len__(self) -> int:
        return self.output.shape[0]

    def __repr__(self) -> str:
        output = self.__class__.__name__  + " \n"
        output += "output =  " + str(self.output) + " \n"
        output += "time =  " + str(self.time) + " \n"
        return output


if __name__ == "__main__":
    import sines
    import soundfile as sf
    audio,sr = sf.read("/Users/julian/Documents/MATLAB/sounds/greasy.wav")
    sm = sines.Sines(step_size=2,confidence=.9,sr_down=16e3,resolutions=1,cent_threshold=10,window_size=20)
    features = sm(audio,sr)

    tt = features['frame']
    xx = features['frequency'][:,None]
    yy = features['amplitude'][:,None]

    data = Data(y=yy,time=tt,x=xx)
    data.load(y=yy,time=tt,x=xx)

    data.plot()
