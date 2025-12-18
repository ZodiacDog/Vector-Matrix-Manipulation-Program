"""
VecMatrix - Advanced Vector and Matrix Manipulation Library
============================================================

A comprehensive library for vector and matrix operations with dual
numerical (NumPy) and symbolic (SymPy) computation modes.

Basic usage:
    >>> from vecmatrix import Vector, Matrix
    >>> v1 = Vector([1, 2, 3])
    >>> v2 = Vector([4, 5, 6])
    >>> v1.dot(v2)
    32.0
    
    >>> A = Matrix([[1, 2], [3, 4]])
    >>> A.det()
    -2.0

Author: ML
Version: 2.0.0
License: MIT
"""

from .core import (
    # Main classes
    Vector,
    Matrix,
    
    # Exceptions
    VecMatrixError,
    ModeError,
    DimensionError,
    SingularMatrixError,
    
    # Utility functions
    tensor_product,
    gram_schmidt,
    matrix_power,
    matrix_exp,
    matrix_log,
)

__version__ = '2.0.0'
__author__ = 'ML'
__license__ = 'MIT'

__all__ = [
    # Classes
    'Vector',
    'Matrix',
    
    # Exceptions
    'VecMatrixError',
    'ModeError',
    'DimensionError',
    'SingularMatrixError',
    
    # Utilities
    'tensor_product',
    'gram_schmidt',
    'matrix_power',
    'matrix_exp',
    'matrix_log',
]
