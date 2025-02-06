# Accelerating Python

## Introduction

This set of tutorials provides a brief introduction two packages, Numba and
JAX, for accelerating Python. The purpose of the tutorial is to make students
aware of the possibilities for achieving high performance and productivity in
Python, without switching to an alternative compiled language.

The rendered version of the book can be found
[here](https://jhale.github.io/accelerating-python).

This material was presented at the first MATHCODA Annual Training Workshop held
in Luxembourg in early 2025.

## Funding

This research was funded in whole, or in part, by the Luxembourg National
Research Fund (FNR), grant reference PRIDE/21/16747448/MATHCODA.

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
Colaboratory](https://colab.research.google.com).

## Syllabus

### Session 1: Introduction to Python acceleration tools

This course will equip candidates to optimize performance critical numerical
computations with modern Python acceleration techniques.

By the end of the course, participants will:

* Understand the strengths, use cases and technology behind tools like Numba
  and JAX.  
* Be able to optimize Python code using JIT compilation.
* Understand the potential for using automatic differentiation in any area of
  science where gradient-based algorithms are used for training, inference,
  calibration, optimisation etc.

### Session 1: Introduction to Python acceleration tools 

* Challenges in Python performance for computational tasks.  
* Overview of Python acceleration tools:  
  * Numba for just-in-time (JIT) compilation.  
  * JAX for automatic differentiation and GPU/TPU acceleration.  

### Session 2: Acceleration with Numba

* Basics of JIT compilation with Numba: `@jit` and `@njit`.  
* Working with Numba-supported NumPy features.  

### Session 3: Acceleration and differentiation with JAX

* Overview of JAX: automatic differentiation and JIT. 
* Using `jax.numpy` as a drop-in replacement for NumPy.  
* Vectorization and parallelism with `vmap`.  
* JIT compilation with `@jit` in JAX.
