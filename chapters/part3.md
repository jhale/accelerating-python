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
programs. This makes it a very powerful tool in any field of science where
gradient-based numerical algorithms are in use, i.e. everywhere, to
fit/train/calibrate highly complex models.

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

```{code-cell}
import jax.numpy as jnp

def f(theta, x):
    mu = theta[0]
    variance = jnp.exp(theta[1])
    n = x.shape[0]
    log_likelihood = -jnp.sum((x - mu)**2)/(2.0 * variance) - 0.5*n*jnp.log(2 * jnp.pi * variance)
    return -log_likelihood
```

We have defined the log-likelihood as a function over three parameters. We
would now like to  fix the likelihood on a particular realisation of the data
`x`. This fixing can be done via partial evaluation. 

```{code-cell}
import numpy as np

mu = 0.5
variance = 2.0
scale = np.sqrt(2.0)
x = np.random.normal(loc=mu, scale=scale, size=100)

from jax.tree_util import Partial
f_x = Partial(f, x=x)
f_x(jnp.array([3.0, 2.0]))
```

:::{note}
One of the peculiarities of JAX is that its transformations can only work on
functions that are *functionally pure*: all input data must be passed through
the function parameters, and all results must be returned through function
results. Furthermore, a pure function will always return the same result if
called with the same inputs. This notion is quite natural and even preferred
for mathematicians, but can require some careful thought when converting more
traditional imperative algorithms which use global state.

Here are some examples of disallowed behaviour.

```{code-block}
g = 0.
def impure_uses_globals(x):
  return x + g

def modifies_globals(x):
  g = x + g
```

JAX does not enforce functional purity, it is up to you!

For further information and examples, see the [JAX
documentation](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions).
:::

### Taking the gradient

We can take the gradient of a scalar-valued function by passing it to
`jax.jacfwd`. This function calculates the gradient using forward-mode
automatic differentiation - we will return to this notion later.

```{code-cell}
import jax

df_x = jax.jacfwd(f_x)
df_x(jnp.array([3.0, 2.0]))
```

### Optimising via gradient descent

We will now find the minimum using gradient descent. We implement the algorithm
in pure Python, and call through to the JAX generated routines.

```{code-cell}
theta = jnp.array([0.0, 0.0])
steps = 100
step_size = 0.01

for _ in range(steps):
    df_x_theta = df_x(theta)
    theta -= step_size * df_x_theta

print(f"Optimal mu: {theta[0]} ({np.mean(x)})")
print(f"Optimal variance: {jnp.exp(theta[1])} ({np.var(x)})")
```

### Optimising via Newton's method

JAX's transformations can be stacked to produce higher-order derivatives. Here
we will apply the forward-mode differentiation twice to produce a function that
can compute the Hessian of $f$.

```{code-cell}
ddf_x = jax.jacfwd(jax.jacfwd(f_x))
ddf_x(jnp.array([3.0, 2.0]))
```

With the Hessian function we can implement a Newton's method.

:::{note}
If we were optimising for only the unknown mean $\mu$ and the variance
$\sigma^2$ was fixed, the function $f$ would be quadratic in $\mu$ - this would
lead to Newton's method converging in a single step.
:::

```{code-cell}
theta = jnp.array([0.0, 0.0])
steps = 10

for _ in range(steps):
    df_x_theta = df_x(theta)
    ddf_x_theta = ddf_x(theta)

    # Solve for the Newton step
    delta_theta = np.linalg.solve(ddf_x_theta, df_x_theta)

    theta -= delta_theta

print(f"Optimal mu: {theta[0]} ({np.mean(x)})")
print(f"Optimal variance: {jnp.exp(theta[1])} ({np.var(x)})")
