# Accelerating Python

## Introduction

This set of tutorials provides a brief introduction to methods for accelerating
Python. This set of tutorials was presented at the first MATHCODA Annual
Training Workshop held in Luxembourg in early 2025.

## Installation

To install the required components for the tutorial:

    python3 -m venv accelerating-python-venv/
    source accelerating-python-venv/bin/activate
    pip install numba numpy
    # macOS only - Apple Metal version.
    pip install jax-metal
    # All other platforms
    pip install jax

The notebooks also run on [Google
Colaboratory](https://colab.research.google.com) using the provided Tensor
Processing Units (TPUs).

## Syllabus

### Session 1: Introduction to Python Acceleration Tools

This course will equip candidates to optimize performance critical numerical
computations with modern Python acceleration techniques.

By the end of the course, participants will:

* Understand the strengths, use cases and technology behind tools like Numba
  and JAX.  
* Be able to optimize Python code using JIT compilation and GPU acceleration.

**Topics Covered:**

* Challenges in Python performance for computational tasks.  
* Overview of Python acceleration tools:  
  * Numba for just-in-time (JIT) compilation.  
  * JAX for automatic differentiation and GPU/TPU acceleration.  
* Brief comparison with other tools (e.g., Cython, PyPy).

**Exercise:**

* Benchmark a simple Python function (e.g., array sum, matrix multiplication)
  to establish a performance baseline.

### Session 2: Accelerating Code with Numba

**Topics Covered:**

* Basics of JIT compilation with Numba: `@jit` and `@njit`.  
* Working with Numba-supported NumPy features.  
* Performance tuning: threading, parallelism, and GPU support.

**Hands-On Exercise:**

* Optimize a computationally intensive Python function (e.g., Mandelbrot set
  generation or numerical integration) using Numba.

### Session 3: JAX for GPU/TPU Acceleration

**Topics Covered:**

* Overview of JAX: automatic differentiation and XLA.  
* Using `jax.numpy` as a drop-in replacement for NumPy.  
* Vectorization and parallelism with `vmap` and `pmap`.  
* JIT compilation with `@jit` in JAX.

**Exercise:**

* Optimize a computationally intensive Python function (e.g., Mandelbrot set
  generation or numerical integration) using JAX.

## Funding

This research was funded in whole, or in part, by the Luxembourg National
Research Fund (FNR), grant reference PRIDE/21/16747448/MATHCODA.
