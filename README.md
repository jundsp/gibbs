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

![DP mixtur model.](https://github.com/jundsp/gibbs/blob/main/examples/mixtures/imgs/gmm_dirichlet_process.png?raw=true)


---
Author: Julian Neri  
Affil: McGill University  
Date: September 2023

