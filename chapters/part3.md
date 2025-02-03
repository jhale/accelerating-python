---
file_format: mystnb
kernelspec:
  name: python3
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: myst
---

# Session 3: Acceleration and differentiation with JAX

JAX is a library for array-oriented numerical computations. Like `numba`, it
includes just-in-time compilation to a wide variety of hardware targets, e.g.
CPU, GPU, TPU etc. 

Perhaps most interestingly, JAX includes automatic differentiation capabilities
that allow it to derive gradients of (nearly) arbitrary numerical Python
programs.

Although JAX is strongly marketed towards machine learning researchers, it is
actually useful in any field of science where gradient-based numerical
algorithms are in use, i.e. everywhere.

## JAX stands in for `numpy`

JAX supports the majority of existing `numpy` functionality through a
stand-in-replacement module `jax.numpy`.

In this example we will perform maximum likelihood estimation on the parameters
of a normal distribution - this is a trivial problem to solve analytically, and
also easy to solve numerically, but the ideas here extend to highly complex
models.

### Problem statement

Assume we have a sequence of iid samples $x_1, \ldots, x_n$ of random variables
assumed to be drawn from a normal distribution with unknown mean $\mu_0$ and
unknown variance $\sigma_0^2$. The negative log-likelihood function is given by

$$
f(\mu, \sigma^2; x_1, \ldots, x_n) =  \frac{n}{2} \ln(\sigma^2) + \frac{n}{2} \ln(2 \pi) + \frac{1}{2\sigma^2} \sum_{j = 1}^{n} (x_j - \mu)^2
$$

Following standard arguments the maximum likelihood estimators of the mean and
variance can be found be found by minimising (analytically or numerically) the
negative log-likelihood. Analytically:

$$
\hat{\mu}_n = \frac{1}{n} \sum_{j = 1}^{n} x_j
$$

$$
\hat{\sigma}_n^2 = \frac{1}{n} \sum_{j = 1}^{n} (x_j - \hat{\mu})^2
$$

Here we will attempt a numerical solution via a gradient-based optimiser and
use JAX's automatic differentiation to derive the necessary gradients. 

### Implementation

```{code-block}
import jax.numpy as jnp
import numpy as np


def log_likelihood(x, mu, variance):
    n = x.shape[0]
    return (1.0/(2.0*variance))*np.sum(x - mu)**2 + (n/2.0)*jnp.log(variance) + (n/2.0)*jnp.log(2.0*pi) 
```

We have defined the log-likelihood as a function over three parameters. We
would now like to generate and fix the likelihood on a particular realisation
of the data `x`. This can 

```{code-block}
mu = 1.0
variance = 1.0
x = np.random.normal(loc=mean, variance=variance, size=100)

from jax.tree_util import Partial
log_likelihood_x = Partial(log_likelihood, x)
print(log_likelihood_x(3.0, 2.0))
```
