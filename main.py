import gibbs
import numpy as np

class Module(object):
    def __init__(self) -> None:
        pass
    def forward(self,data):
        samples = np.random.normal(0,1)
        return samples