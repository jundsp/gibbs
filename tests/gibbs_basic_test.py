import unittest   # The test framework

from gibbs import Gibbs, SLDS, LDS, GHMM, GMM, Data
import numpy as np

def test_sampling(y,model,sampler):
    iters = 10
    for iter in (range(iters)):
        model(y)
        sampler.step(model.named_parameters())
    print("estimating from median of samples")
    sampler.get_estimates(burn_rate=.8)
    return 0

def testing_function_lds(y):
    print("creating lds")
    model = LDS(output_dim=2,state_dim=2)
    sampler = Gibbs()
    print("sampling lds")
    return test_sampling(y,model,sampler)

def testing_function_slds(y):
    print("creating slds")
    model = SLDS(output_dim=2,state_dim=2,states=4)
    sampler = Gibbs()
    print("sampling slds")
    return test_sampling(y,model,sampler)

def testing_function_ghmm(y):
    print("creating ghmm")
    model = GHMM(output_dim=1,states=3)
    sampler = Gibbs()
    print("sampling gmm")
    return test_sampling(y,model,sampler)

def testing_function_gmm(y):
    print("creating gmm")
    model = GMM(output_dim=1,components=3)
    sampler = Gibbs()
    print("sampling gmm")
    return test_sampling(y,model,sampler)


class Test_TestBasic(unittest.TestCase):
    def test_gmm(self):
        y = np.random.normal(0,1,(100,2))
        data = Data(y)
        self.assertEqual(testing_function_gmm(data), 0)

    def test_ghmm(self):
        y = np.random.normal(0,1,(100,1,2))
        self.assertEqual(testing_function_ghmm(y), 0)

    def test_lds(self):
        y = np.random.normal(0,1,(100,2))
        data = Data(y)
        self.assertEqual(testing_function_lds(data), 0)

    def test_slds(self):
        y = np.random.normal(0,1,(100,2))
        data = Data(y)
        self.assertEqual(testing_function_slds(data), 0)

if __name__ == '__main__':
    unittest.main()
