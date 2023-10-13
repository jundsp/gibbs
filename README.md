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
A collection of examples are contained in the "examples" directory. These cover using the Gibbs package to infer a variety of the implemented Bayesian models: mixture models, linear dyanmical systems, switching dynamical systems, hidden Markov models, dirichlet process, and more.

In the mixture examples, we can infer a Dirichlet process mixture model (a Gaussian mixture model with an infinitely countable number of components), as illustrated in the following figure.

![alt text](https://github.com/jundsp/gibbs/blob/main/examples/mixtures/imgs/gmm_dirichlet_process.pdf?raw=true)


---
Author: Julian Neri  
Affil: McGill University  
Date: September 2023

