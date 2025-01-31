## Acceleration with Numba

Numba is a *just-in-time* compiler for a subset of Python code. It is
particularly useful for accelerating numerical algorithms written using
classical imperative techniques, i.e. for loops over arrays of data. As we will
see, standard Python and even numpy-based for loops fail to provide reasonable
performance for this type of algorithm. 

Numba can generate code for CPU targets via LLVM, and also CUDA-compatible GPUs
via the `numba-cuda` package.

### Matrix-matrix multiplication

The matrix-matrix product is a cornerstone of numerical computations such as
standard deep-neural network architectures.

Let $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n
\times p}$ be two matrices. Their product \mathbf{C} \in \mathbb{R}^{is given by:

$$
\mathbf{C} = \mathbf{A} \mathbf{B}
$$

where $\mathbf{C} \in \mathbb{R}^{m \times p}$ and its entries are computed as:

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

### numpy range-based for loops 


