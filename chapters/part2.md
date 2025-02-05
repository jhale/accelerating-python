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
relatively standard numerical Python functions and compile them into optimised
low-level machine code.

Numba is particularly useful for accelerating numerical algorithms written
using classical techniques, i.e. for loops over arrays of data. Standard Python
(not shown) and numpy-based for loops do not provide reasonable performance for
this type of algorithm. This makes numba a good tool for re-writing intensive
parts of Python programs without ever leaving Python.

:::{note}
Numba can generate code for CPU targets via LLVM, and also CUDA-compatible GPUs
via the `numba-cuda` package. Not that targetting CUDA typically requires
substantial changes to the standard Python code, unlike the CPU target.
:::

:::{note}
Recent work on `numba-enzyme` brings automatic differentiation capabilities to
Numba and also composability with JAX, which we will see in Part 3.
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

n = 128
A = np.array(rng.random((n, n)), order="C")
B = np.array(rng.random((n, n)), order="F")

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
array $\mathbf{b}$ with $n \times m$ entries $\left\lbrace b_1, b_2, \ldots,
b_{n \times m} \right\rbrace$.

So interpret the entries of $\mathbf{a}$ as entries of $\mathbf{b}$ we need a
map between the two.

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
to iterate over the entries of $\mathbf{A}$ and $\mathbf{B}$ *contiguously*,
that is, the order they are stored in memory.

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
        for k in range(n):  # Iterate over common dimension
            for j in range(p):  # Iterate over columns of B
                C[i, j] += A[i, k] * B[k, j]

    return C

%timeit C_loop = matrix_multiply_loops(A, B)
```

On my computer this implementation is 4 orders of magnitude slower than the
BLAS-provided version. The main (although not only) reasons for this is that
Python:

- is an interpreted language, which means code is executed line by line, rather
  than being transformed/compiled to machine code before execution.
- is dynamically typed, which means types are determined and checked at
  runtime.

This adds massive overheads. Using numpy gets around Python's limitations by
immediately dispatching `np.dot(A, B)` to a compiled BLAS routine, typically
written in C, Fortran or even assembly.

## Accelerating with numba

Numba can just-in-time compile most Python code into highly optimised machine
code. It is particularly effective compiling code with 'tight loops' like the
matrix matrix multiplcation we've just seen. Numba compiles in a sequence of
steps, where it transforms Python code to the Numba intermediate
representation, and then it transforms this into LLVM assembly code which can
represent all high-level languages cleanly. The LLVM compiler then transforms
LLVM to machine code.

Although the underlying transformations are complex, using `numba` is
relatively straighforward. We just-in-time compile our function by applying the
`@jit` decorator:

```{code-cell}
import numba

@numba.jit
def matrix_multiply_loops_jit(A, B):
    m, n = A.shape
    n2, p = B.shape
    
    if n != n2:
        raise ValueError("Inner matrix dimensions must agree.")

    C = np.zeros((m, p))

    for i in range(m):
        for k in range(n):
            for j in range(p):
                C[i, j] += A[i, k] * B[k, j]

    return C

# 'Warm up' - calling the function once starts Numba's JIT process.
C_jit = matrix_multiply_loops_jit(A, B)
assert np.allclose(C_jit, C_dot) 
%timeit C_jit = matrix_multiply_loops_jit(A, B)
```

On my system this is around three times faster than the highly optimised macOS
BLAS - this is not bad at all, given this is far from an optimal
implementation, which require the use of more complex techniques such as
blocking, parallelisation and explicit use of special CPU instructions.

### Further optimisations

#### `nopython` mode

In its default mode `@jit` Numba can compile an extensive subset of Python
code. With `@jit(nopython=False)` Numba can compile almost *all* Python code.
But this comes at a cost -- Numba will call back into the Python intepreter
when it finds Python code that it cannot translate. My anecdotal experience is
that `@jit(nopython=False)` rarely provides better performance than pure
Python.

:::{note}
In versions of Numba prior to 0.59 it was necessary to explicitly use
`@jit(nopython=True)` or equivalently `@njit` - this is no longer the case, and
just `@jit` is recommended and sufficient.
:::

#### `fastmath` mode

LLVM produces IEEE754 compliant floating point arithmetic operations by
default. IEEE754 is strict and generally produces good results if you use
reasonable algorithms designed for stability and accuracy, but relaxing these
rules can lead to better performance. Enabling `fastmath` gives LLVM the
opportunity to ignore IEEE754 specification and produce faster code, but be
aware it can lead to subtle numerical issues.

One optimisation included with `fastmath` is turning off associative math. This
allows the compiler to re-associate operands in a series of floating point
operations.

$$
f(a, b, c) = a + b + c
$$
In real arithmetic, any ordering of the operations in `f` is mathematically
equivalent.

However, in floating point arithmetic, if we know that `a` is likely to be very
large, and `b` very small, and `c` moderately sized, we might choose to write
the function as:
```{code-cell}
@numba.jit()
def f(a, b, c):
    return (a + b) + c
```
With `fastmath`, LLVM could decide to re-arrange this internally to:
```{code-cell}
@numba.jit()
def f_fast(a, b, c):
    return a + (b + c)
```

```{code-cell}
a = np.array([1e9])
b = np.array([-1e9])
c = np.array([0.1])

print(f"f: {f(a, b, c)[0]:.12f}")
print(f"f_fast: {f_fast(a, b, c)[0]:.12f}")
```

My advice is to always develop code without `fastmath` and only turn it on
after thoroughly testing for correctness and performance. 

#### Advanced: Signatures and inspecting generated assembly

When calling a function decorated with `@jit`, numba will inspect the arguments
to the function and compile a specialised version for those arguments. We can
view the *specialisations* generated by Numba using:

```{code-cell}
matrix_multiply_loops_jit.signatures
```

It is possible to inspect the transformations that numba produces from Python,
to Numba IR, to LLVM and finally to assembly.

Numba IR:

```{code-cell}
:tags: ["scroll-output"]
print(matrix_multiply_loops_jit.inspect_types(signature=matrix_multiply_loops_jit.signatures[0]))
```

LLVM:

```{code-cell}
:tags: ["scroll-output"]
print(matrix_multiply_loops_jit.inspect_llvm(signature=matrix_multiply_loops_jit.signatures[0]))
```

Assembly:

```{code-cell}
:tags: ["scroll-output"]
print(matrix_multiply_loops_jit.inspect_asm(signature=matrix_multiply_loops_jit.signatures[0]))
```

### Exercises

1. Run an experiment where you increase the size of the square matrices passed
   to `A@B` and `matrix_multiply_loops_jit` in a geometric sequence starting at
   4 -- you should be able to go up to around 1024. Plot the results using
   `matplotlib`. `%timeit -o` will place the timings into a variable.

```{code-cell}
result = %timeit -o matrix_multiply_loops_jit(A, B)
result.average
```

You will need to execute the function inside a loop and store the results in
arrays.

```
ns = np.geomspace(...)
timings1 = []
timings2 = []
for n in np.geomspace(...):
    # Your code
    pass
```

2. For a fixed sized matrices of reasonable size, try passing combinations of
   different orderings to `matrix_multiply_loops`. What do you observe?


### References and further information

- [The Numba documentation](https://numba.pydata.org/numba-doc/dev/index.html)
- [Extensions to CUDA](https://tbetcke.github.io/hpc_lecture_notes/gpu_introduction.html)
