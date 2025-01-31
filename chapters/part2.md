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

## Session 2: Acceleration with Numba

Numba is a *just-in-time* compiler for a subset of Python code. It can take
relatively standard Python programs and compile them into optimised low-level
machine code.

Numba is particularly useful for accelerating numerical algorithms written using
classical imperative techniques, i.e. for loops over arrays of data. As we will
see, standard Python and even numpy-based for loops fail to provide reasonable
performance for this type of algorithm. 

Numba can generate code for CPU targets via LLVM, and also CUDA-compatible GPUs
via the `numba-cuda` package. Recent work on `numba-enzyme` brings automatic
differentiation capabilities to Numba and also composability with JAX.

### Matrix-matrix multiplication

The matrix-matrix product is a cornerstone of numerical computations such as
standard deep-neural network architectures.

Let $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n
\times p}$ be two matrices. Their product is $\mathbf{C} \in \mathbb{R}^{m
\times p}$ := \mathbf{A} \mathbf{B} is given with entries
$$
C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}, \quad \text{for } 1 \leq i \leq m, \ 1 \leq j \leq p.
$$

In practice, you should almost never write your own routine to perform
matrix-matrix product - BLAS Level 3 will provide an optimised version for your
specific hardware. However, this is a good example to show the basic use of
numba.

The naive algorithm will scale at $O(mnk)$.

### numpy built-in (BLAS Level 3)

Numpy will call the BLAS Level 3 routine when you call the high-level `np.dot`
or `A@B`.

```{code-cell}
import numpy as np
```

### numpy range-based for loops 

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
```
