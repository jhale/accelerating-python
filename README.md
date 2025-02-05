# Accelerating Python

## Introduction

This set of tutorials provides a brief introduction to methods for accelerating
Python and is aimed at first year graduate students and PhDs. 

This material was presented at the first MATHCODA Annual Training Workshop held
in Luxembourg in early 2025.

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

### Session 1: Introduction to Python acceleration tools

This course will equip candidates to optimize performance critical numerical
computations with modern Python acceleration techniques.

By the end of the course, participants will:

* Understand the strengths, use cases and technology behind tools like Numba
  and JAX.  
* Be able to optimize Python code using JIT compilation and GPU acceleration.

### Session 1: Introduction to Python acceleration tools 

* Challenges in Python performance for computational tasks.  
* Overview of Python acceleration tools:  
  * Numba for just-in-time (JIT) compilation.  
  * JAX for automatic differentiation and GPU/TPU acceleration.  

### Session 2: Acceleration with Numba

**Topics Covered:**

* Basics of JIT compilation with Numba: `@jit` and `@njit`.  
* Working with Numba-supported NumPy features.  

### Session 3: Acceleration and differentiation with JAX

**Topics Covered:**

* Overview of JAX: automatic differentiation and JIT. 
* Using `jax.numpy` as a drop-in replacement for NumPy.  
* Vectorization and parallelism with `vmap`.  
* JIT compilation with `@jit` in JAX.

## Funding

This research was funded in whole, or in part, by the Luxembourg National
Research Fund (FNR), grant reference PRIDE/21/16747448/MATHCODA.
