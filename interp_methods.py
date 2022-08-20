from math import pi
import numba
from numba import jit
from numba import njit

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy
except ImportError:
    numpy = None

if numpy is None and torch is None:
    raise ImportError("Must have either Numpy or PyTorch but both not found")

def set_framework_dependencies(x):
    if type(x) is numpy.ndarray:
        to_dtype = lambda a: a
        fw = numpy
    else:
        to_dtype = lambda a: a.to(x.dtype)
        fw = torch
    eps = fw.finfo(fw.float32).eps
    return fw, to_dtype, eps

@jit(fastmath=True,forceobj=True,cache=True)
def support_sz(sz):
    def wrapper(f):
        f.support_sz = sz
        return f
    return wrapper


@njit(parallel=True,fastmath=True,cache=True)
def cubic(x):
    fw,to_dtype,eps=set_framework_dependencies(x)
    absx=fw.abs(x)
    absx2=absx ** 2
    absx3=absx ** 3
    return ((1.5*absx3-2.5*absx2+1.0)*to_dtype(absx<=1.0)+(-0.5*absx3+2.5*absx2-4.0*absx+2.0)*to_dtype((1.0<absx)&(absx<=2.0)))


@jit(fastmath=True,forceobj=True,cache=True)
def lanczos2(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return (((fw.sin(pi * x) * fw.sin(pi * x / 2) + eps) /
            ((pi**2 * x**2 / 2) + eps)) * to_dtype(abs(x) < 2))


@jit(fastmath=True,forceobj=True,cache=True)
def lanczos3(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return (((fw.sin(pi * x) * fw.sin(pi * x / 3) + eps) /
            ((pi**2 * x**2 / 3) + eps)) * to_dtype(abs(x) < 3))


@jit(fastmath=True,forceobj=True,cache=True)
def linear(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return ((x + 1) * to_dtype((-1 <= x) & (x < 0)) + (1 - x) *
            to_dtype((0 <= x) & (x <= 1)))


@jit(fastmath=True,forceobj=True,cache=True)
def box(x):
    fw, to_dtype, eps = set_framework_dependencies(x)
    return to_dtype((-1 <= x) & (x < 0)) + to_dtype((0 <= x) & (x <= 1))
