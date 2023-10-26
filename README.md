# gibbs

A python package for Gibbs sampling of Bayesian hierarchical models.

Includes base classes for sampling and modules for a variety of popular Bayesian models like time-series, finite, and infinite mixture models.

## Installation

Clone the repository, and enter the directory,
```console
cd /gibbs
```
Then install the package,
```console
python3 -m pip install .
```

---
## Unit tests
To run the unit test from root directory:

```console 
python -m unittest tests/gibbs_basic_test.py
```

---
## Examples
A collection of examples are contained in the "examples" directory. These cover using the Gibbs package to infer a variety of the implemented Bayesian models: mixture models, hidden Markov models, linear dynamical systems, switching dynamical systems, dirichlet processes, and more.

### Dirichlet process mixture model
In one of the mixture examples, we fit a Dirichlet process mixture model (a Gaussian mixture model with an infinitely countable number of components) from 2D data. This performs unsupervised classification of the data. The inferred components (2D Gaussians) and clustered data are depicted in the following figure.

```python
import gibbs
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('gibbs.mplstyles.latex')

# Generate some 2D GMM data.
np.random.seed(8)
y, z = gibbs.gmm_generate(500,2,5)  
data = gibbs.Data(y=y) # Creates data object to be used by Gibbs.

```

<!-- ```python
# Plot the data
figsize=(3,2.5)
colors = gibbs.get_colors()
data.plot()
gibbs.scattercat(data.output,z,figsize=figsize,colors=colors)
plt.show()

``` -->

```python
# Creates the model / sampler
model = gibbs.InfiniteGMM(collapse_locally=True,sigma_ev=1) # Create model
sampler = gibbs.Gibbs() # Get the base sampler

```

```python
# Fitting model to data
sampler.fit(data,model,samples=50) # Fit model to data with 50 samples

# Retrieve the samples and compute expected value
chain = sampler.get_chain(burn_rate=.9,flatten=False) # Get the sample chain
z_hat = gibbs.categorical2multinomial(chain['z']).mean(0).argmax(-1) # Compute expected value

```

```python
# Plot results
gibbs.scattercat(data.output,z_hat,figsize=figsize,colors=colors)
for k in np.unique(z_hat):
    idx = z_hat == k
    mu,S,nu = model._predictive_parameters(*model._posterior(model.y[idx],*model.theta))
    cov = S * (nu)/(nu-2)
    gibbs.plot_cov_ellipse(mu,cov,fill=None,color=colors[k])
plt.show()
```


![DP mixture model.](https://github.com/jundsp/gibbs/blob/main/examples/mixtures/imgs/gmm_dirichlet_process.png?raw=true)

### Switching Linear Dynamical System

Switching linear dynamical systems are temporal models that have both a discrete state (HMM) and a continuous state (LDS/Kalman filter).
In the example "slds_ex.py" in the examples folder, an SLDS is fit to 1D time-series data.
The data is a sinusoidal oscillation that has discrete changes in frequency.

![SLDS.](https://github.com/jundsp/gibbs/blob/main/examples/slds/imgs/slds_ex.png?raw=true)

![SLDS sample chain for the discrete state.](https://github.com/jundsp/gibbs/blob/main/examples/slds/imgs/slds_ex_chain.png?raw=true)

---
Author: Julian Neri  
Affil: McGill University  
Date: September 2023

