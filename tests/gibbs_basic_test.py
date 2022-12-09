import unittest   # The test framework

from gibbs import Gibbs, SLDS, LDS
import numpy as np

def testing_function_lds(y):
    print("creating slds")
    model = LDS(output_dim=1,state_dim=2)
    sampler = Gibbs()

    print("sampling lds")
    iters = 10
    for iter in (range(iters)):
        model(y)
        sampler.step(model.named_parameters())

    print("estimating from median of samples")
    sampler.get_estimates(burn_rate=.8)
    return 0

def testing_function_slds(y):
    print("creating slds")
    model = SLDS(output_dim=1,state_dim=2,states=4)
    sampler = Gibbs()

    print("sampling slds")
    iters = 10
    for iter in (range(iters)):
        model(y)
        sampler.step(model.named_parameters())

    print("estimating from median of samples")
    sampler.get_estimates(burn_rate=.8)
    return 0


class Test_TestBasic(unittest.TestCase):
    def test_lds(self):
        y = np.random.normal(0,1,(100,2))
        self.assertEqual(testing_function_lds(y), 0)

    def test_slds(self):
        y = np.random.normal(0,1,(100,2))
        self.assertEqual(testing_function_slds(y), 0)

if __name__ == '__main__':
    unittest.main()
