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

# Session 2: Acceleration with Numba

Numba is a *just-in-time* compiler for a subset of Python code. It can take
relatively standard numerical Python programs and compile them into optimised
low-level machine code.

Numba is particularly useful for accelerating numerical algorithms written
using classical imperative techniques, i.e. for loops over arrays of data.
Standard Python (not shown) and numpy-based for loops do not provide
reasonable performance for this type of algorithm. 

:::{note}
Numba can generate code for CPU targets via LLVM, and also CUDA-compatible GPUs
via the `numba-cuda` package. Recent work on `numba-enzyme` brings automatic
differentiation capabilities to Numba and also composability with JAX.
:::

## Matrix-matrix multiplication

The matrix-matrix product is a fundamental operation in numerical models
including deep-neural networks, regressions and solvers for partial
differential equations. 

Let $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n
\times p}$ be two matrices. Their product is $\mathbf{C} := \mathbf{A}
\mathbf{B} \in \mathbb{R}^{m \times p}$ has entries
:::{math}
C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}, \quad \text{for } 1 \leq i \leq m, \ 1 \leq j \leq p.
:::

Given $\mathbf{A}$ and $\mathbf{B}$ a simple algorithm in pseudo-code to
calculate $C$ is

```
for i from 0 to m - 1:         // Iterate over rows of A
    for j from 0 to p - 1:     // Iterate over columns of B
        for k from 0 to n - 1: // Iterate over shared dimension
            C[i, j] ← C[i, j] + A[i, k] * B[k, j]
```

This basic algorithm will scale at $O(mnk)$, or $O(n^3)$ for square matrices
with equal dimensions.

## Numpy built-in via BLAS

In practice, you should almost never write your own routine to perform
matrix-matrix product - BLAS Level 3 will provide an optimised version for your
specific hardware. However, this is a good example to show the basic use of
numba.

Numpy will call the BLAS Level 3 GEMM routine when you call the high-level
`np.dot` or `A@B`.

```{code-cell}
import numpy as np
rng = np.random.default_rng()

A = np.array(rng.random((400, 1000)), order="C")
B = np.array(rng.random((1000, 700)), order="F")

%timeit C_dot = np.dot(A, B)
```

The scipy package also has the ability to call precise BLAS routines
```{code-cell}
import scipy as sp
gemm = sp.linalg.get_blas_funcs("gemm", arrays=(A, B))
%timeit C_gemm = gemm(1.0, A, B)
```

This result can be considered the best-case scenario. Let's check that
the results of both routines are the same to machine precision

```{code-cell}
C_dot = np.dot(A, B)
C_gemm = gemm(1.0, A, B)
assert np.allclose(C_dot, C_gemm)
```

:::{note} 
A computer's memory can be viewed as being logically flat. Knowing this is
important for performance when we think about writing algorithms that move over
tensors (matrices) with two or more dimensions.

Consider a matrix $\mathbf{a} \in \mathbb{R}^{n \times m}$ with $n \times m$
entries $a_{ij}$. In the computer's memory the entries must be stored as a flat
array $\left\lbrace b_1, b_2, \ldots, b_{n \times m} \right\rbrace$.

So, we must have a map between the entries of the two-dimensional matrix
$\mathbf{a}$ and the entries in memory $\mathbf{b}$.

The two logical ways to do this are *row-major* and *column-major* ordering.
For row-major ordering, which is used by default in Numpy and C:

$$
(i, j) \to in + j
$$

and column-major ordering, which is used by default in Fortran and MATLAB:

$$
(i, j) \to jm + i
$$

```{figure} ordering.svg
:align: center
:width: 50%

Column ordering vs row-ordering for a $3 \times 3$ matrix $\mathbf{a}$. By
Cmglee - CC BY-SA 4.0,
[Original](https://commons.wikimedia.org/w/index.php?curid=65107030)
```

When writing the matrix-matrix multiplication it is important for performance
to iterate over the entries of $\mathbf{A}$ and $\mathbf{B}$ linearly in
memory.

```
for i from 0 to m - 1:         // Iterate over rows of A
    for k from 0 to n - 1:     // Iterate over shared dimension
        for j from 0 to p - 1: // Iterate over columns of B
            C[i, j] ← C[i, j] + A[i, k] * B[k, j]
```

With this version of the algorithm it is optimal to store `A` in row-major
order, `B` in column-major order, and the result `C` in row-major order.
:::


## numpy range-based for loops 

We can achieve the same result by implementing the basic algorithm using numpy
arrays and range-based for loops. The code looks very similar to the
pseudo-code.

```{code-cell}
def matrix_multiply_loops(A, B):
    """
    Perform matrix-matrix multiplication using explicit iteration.
    
    Parameters:
    A: First matrix of shape (m, n)
    B: Second matrix of shape (n, p)
    
    Returns:
      Result matrix of size (m, p)
    """
    m, n = A.shape
    n2, p = B.shape
    
    # Ensure the matrices have compatible dimensions
    if n != n2:
        raise ValueError("Inner matrix dimensions must agree.")

    # Initialize result matrix with zeros
    C = np.zeros((m, p))

    for i in range(m):  # Iterate over rows of A
        for j in range(p):  # Iterate over columns of B
            for k in range(n):  # Iterate over common dimension
                C[i, j] += A[i, k] * B[k, j]

    return C

C_loop = matrix_multiply_loops(A, B)
```
