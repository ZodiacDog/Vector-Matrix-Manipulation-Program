"""
VecMatrix Core Library
======================

A comprehensive, production-grade vector and matrix manipulation library
supporting both numerical (NumPy) and symbolic (SymPy) computation modes.

Features:
- Dual-mode operation (numerical/symbolic)
- Comprehensive linear algebra operations
- Automatic mode detection
- Type safety and validation
- Performance optimized
- Thread-safe operations

Author: ML
Version: 2.0.0
License: MIT
"""

import sys
import numpy as np
from scipy import linalg
import sympy as sp
from typing import Union, List, Tuple, Optional, Any
from functools import wraps
import threading

# Type aliases
NumericType = Union[int, float, complex]
VectorData = Union[List[NumericType], np.ndarray, sp.Matrix]
MatrixData = Union[List[List[NumericType]], np.ndarray, sp.Matrix]


class VecMatrixError(Exception):
    """Base exception for VecMatrix library"""
    pass


class ModeError(VecMatrixError):
    """Raised when operations between different modes are attempted"""
    pass


class DimensionError(VecMatrixError):
    """Raised when dimension requirements are not met"""
    pass


class SingularMatrixError(VecMatrixError):
    """Raised when attempting to invert a singular matrix"""
    pass


def validate_mode(func):
    """Decorator to validate mode compatibility between objects"""
    @wraps(func)
    def wrapper(self, other, *args, **kwargs):
        if hasattr(other, 'mode') and self.mode != other.mode:
            raise ModeError(
                f"Cannot perform {func.__name__} between {self.mode} and {other.mode} modes. "
                f"Convert one to match using .to_numerical() or .to_symbolic()"
            )
        return func(self, other, *args, **kwargs)
    return wrapper


def thread_safe(func):
    """Decorator to make operations thread-safe"""
    lock = threading.RLock()
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)
    return wrapper


class Vector:
    """
    A versatile Vector class supporting numerical (numpy) and symbolic (sympy) operations.
    
    Attributes:
        data: The underlying vector data (numpy array or sympy Matrix)
        mode: 'numerical' or 'symbolic'
        shape: Tuple representing vector dimensions
    
    Examples:
        >>> v1 = Vector([1, 2, 3])
        >>> v2 = Vector([4, 5, 6])
        >>> v1.dot(v2)
        32.0
        
        >>> v_sym = Vector(['x', 'y', 'z'], symbolic=True)
        >>> v_sym.norm()
        sqrt(x**2 + y**2 + z**2)
    """
    
    def __init__(self, data: VectorData, symbolic: bool = False):
        """
        Initialize a Vector.
        
        Args:
            data: Vector data as list, numpy array, or sympy Matrix
            symbolic: Force symbolic mode (auto-detected if data contains symbols)
        
        Raises:
            ValueError: If data is empty or invalid
        """
        if not self._validate_data(data):
            raise ValueError("Vector data cannot be empty")
        
        # Auto-detect symbolic data
        if not symbolic and self._is_symbolic_data(data):
            symbolic = True
        
        if symbolic:
            self.data = sp.Matrix(data)
            self.mode = 'symbolic'
        else:
            self.data = np.array(data, dtype=np.complex128)
            self.mode = 'numerical'
        
        self.shape = self.data.shape if self.mode == 'symbolic' else self.data.shape
    
    @staticmethod
    def _validate_data(data: Any) -> bool:
        """Validate that data is non-empty"""
        if isinstance(data, (list, tuple)):
            return len(data) > 0
        elif isinstance(data, np.ndarray):
            return data.size > 0
        elif isinstance(data, sp.Matrix):
            return data.shape[0] > 0
        return False
    
    @staticmethod
    def _is_symbolic_data(data: Any) -> bool:
        """Detect if data contains symbolic expressions"""
        try:
            flat = np.array(data).flatten()
            return any(isinstance(x, (sp.Expr, sp.Symbol, sp.Basic)) or 
                      isinstance(x, str) for x in flat)
        except:
            return False
    
    def __str__(self) -> str:
        if self.mode == 'numerical':
            # Format complex numbers nicely
            return np.array2string(self.data, precision=6, suppress_small=True)
        return str(self.data)
    
    def __repr__(self) -> str:
        return f"Vector({self.data}, mode='{self.mode}', shape={self.shape})"
    
    def __len__(self) -> int:
        """Return the length of the vector"""
        return self.shape[0] if isinstance(self.shape, tuple) else len(self.data)
    
    def __getitem__(self, index: int):
        """Get element at index"""
        return self.data[index]
    
    def __setitem__(self, index: int, value):
        """Set element at index"""
        self.data[index] = value
    
    @validate_mode
    def add(self, other: 'Vector') -> 'Vector':
        """Add two vectors"""
        self._check_dimensions(other, "addition")
        return Vector(self.data + other.data, symbolic=(self.mode == 'symbolic'))
    
    @validate_mode
    def subtract(self, other: 'Vector') -> 'Vector':
        """Subtract two vectors"""
        self._check_dimensions(other, "subtraction")
        return Vector(self.data - other.data, symbolic=(self.mode == 'symbolic'))
    
    def scale(self, scalar: NumericType) -> 'Vector':
        """Multiply vector by a scalar"""
        return Vector(self.data * scalar, symbolic=(self.mode == 'symbolic'))
    
    @validate_mode
    def dot(self, other: 'Vector') -> Union[float, complex, sp.Expr]:
        """Compute dot product with another vector"""
        self._check_dimensions(other, "dot product")
        if self.mode == 'numerical':
            return np.vdot(self.data, other.data)
        else:
            return self.data.dot(other.data)
    
    def norm(self, ord: Optional[Union[int, float, str]] = None) -> Union[float, sp.Expr]:
        """
        Compute vector norm.
        
        Args:
            ord: Order of the norm (None=2-norm, 1, 2, inf, -inf, 'fro')
        """
        if self.mode == 'numerical':
            return np.linalg.norm(self.data, ord=ord)
        else:
            if ord is None or ord == 2:
                return sp.sqrt(self.data.dot(self.data))
            elif ord == 1:
                return sum(sp.Abs(x) for x in self.data)
            elif ord == float('inf'):
                return sp.Max(*[sp.Abs(x) for x in self.data])
            else:
                raise ValueError(f"Symbolic mode only supports ord=None, 1, 2, or inf")
    
    @validate_mode
    def cross(self, other: 'Vector') -> 'Vector':
        """Compute cross product (3D vectors only)"""
        if len(self) != 3 or len(other) != 3:
            raise DimensionError("Cross product only defined for 3D vectors")
        
        if self.mode == 'numerical':
            return Vector(np.cross(self.data, other.data), symbolic=False)
        else:
            return Vector(self.data.cross(other.data), symbolic=True)
    
    def normalize(self) -> 'Vector':
        """Return unit vector in same direction"""
        norm = self.norm()
        if self.mode == 'numerical' and np.isclose(norm, 0):
            raise ValueError("Cannot normalize zero vector")
        if self.mode == 'symbolic' and norm == 0:
            raise ValueError("Cannot normalize zero vector")
        return Vector(self.data / norm, symbolic=(self.mode == 'symbolic'))
    
    @validate_mode
    def project_onto(self, other: 'Vector') -> 'Vector':
        """Project this vector onto another vector"""
        return other.scale(self.dot(other) / other.dot(other))
    
    @validate_mode
    def angle_between(self, other: 'Vector') -> Union[float, sp.Expr]:
        """Compute angle between two vectors (in radians)"""
        cos_angle = self.dot(other) / (self.norm() * other.norm())
        if self.mode == 'numerical':
            return np.arccos(np.clip(cos_angle, -1.0, 1.0))
        else:
            return sp.acos(cos_angle)
    
    @validate_mode
    def distance_to(self, other: 'Vector') -> Union[float, sp.Expr]:
        """Compute Euclidean distance to another vector"""
        return self.subtract(other).norm()
    
    def is_orthogonal_to(self, other: 'Vector', tol: float = 1e-10) -> bool:
        """Check if this vector is orthogonal to another"""
        dot_prod = self.dot(other)
        if self.mode == 'numerical':
            return abs(dot_prod) < tol
        else:
            return dot_prod.simplify() == 0
    
    def to_symbolic(self) -> 'Vector':
        """Convert to symbolic mode"""
        if self.mode == 'symbolic':
            return self
        return Vector(sp.Matrix(self.data), symbolic=True)
    
    def to_numerical(self) -> 'Vector':
        """Convert to numerical mode"""
        if self.mode == 'numerical':
            return self
        try:
            # Try to convert symbolic to numerical
            data = np.array([complex(x) for x in self.data], dtype=np.complex128)
            return Vector(data, symbolic=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot convert symbolic vector to numerical: {e}")
    
    def to_list(self) -> List:
        """Convert to Python list"""
        if self.mode == 'numerical':
            return self.data.tolist()
        return [x for x in self.data]
    
    def _check_dimensions(self, other: 'Vector', operation: str):
        """Check dimension compatibility"""
        if len(self) != len(other):
            raise DimensionError(
                f"Cannot perform {operation} on vectors of different lengths: "
                f"{len(self)} vs {len(other)}"
            )


class Matrix:
    """
    A versatile Matrix class supporting numerical (numpy) and symbolic (sympy) operations.
    
    Attributes:
        data: The underlying matrix data (numpy array or sympy Matrix)
        mode: 'numerical' or 'symbolic'
        shape: Tuple representing matrix dimensions (rows, cols)
    
    Examples:
        >>> m = Matrix([[1, 2], [3, 4]])
        >>> m.det()
        -2.0
        
        >>> m_sym = Matrix([['a', 'b'], ['c', 'd']], symbolic=True)
        >>> m_sym.det()
        a*d - b*c
    """
    
    def __init__(self, data: MatrixData, symbolic: bool = False):
        """
        Initialize a Matrix.
        
        Args:
            data: Matrix data as nested list, numpy array, or sympy Matrix
            symbolic: Force symbolic mode (auto-detected if data contains symbols)
        
        Raises:
            ValueError: If data is empty or invalid
        """
        if not self._validate_data(data):
            raise ValueError("Matrix data cannot be empty")
        
        # Auto-detect symbolic data
        if not symbolic and self._is_symbolic_data(data):
            symbolic = True
        
        if symbolic:
            self.data = sp.Matrix(data)
            self.mode = 'symbolic'
        else:
            self.data = np.array(data, dtype=np.complex128)
            self.mode = 'numerical'
        
        self.shape = self.data.shape
    
    @staticmethod
    def _validate_data(data: Any) -> bool:
        """Validate that data is non-empty"""
        if isinstance(data, (list, tuple)):
            return len(data) > 0 and len(data[0]) > 0
        elif isinstance(data, np.ndarray):
            return data.size > 0
        elif isinstance(data, sp.Matrix):
            return data.shape[0] > 0 and data.shape[1] > 0
        return False
    
    @staticmethod
    def _is_symbolic_data(data: Any) -> bool:
        """Detect if data contains symbolic expressions"""
        try:
            flat = np.array(data).flatten()
            return any(isinstance(x, (sp.Expr, sp.Symbol, sp.Basic)) or 
                      isinstance(x, str) for x in flat)
        except:
            return False
    
    @classmethod
    def identity(cls, n: int, symbolic: bool = False) -> 'Matrix':
        """Create an identity matrix of size n x n"""
        if symbolic:
            return cls(sp.eye(n), symbolic=True)
        return cls(np.eye(n), symbolic=False)
    
    @classmethod
    def zeros(cls, rows: int, cols: int, symbolic: bool = False) -> 'Matrix':
        """Create a zero matrix"""
        if symbolic:
            return cls(sp.zeros(rows, cols), symbolic=True)
        return cls(np.zeros((rows, cols)), symbolic=False)
    
    @classmethod
    def ones(cls, rows: int, cols: int, symbolic: bool = False) -> 'Matrix':
        """Create a matrix of ones"""
        if symbolic:
            return cls(sp.ones(rows, cols), symbolic=True)
        return cls(np.ones((rows, cols)), symbolic=False)
    
    @classmethod
    def random(cls, rows: int, cols: int, low: float = 0, high: float = 1) -> 'Matrix':
        """Create a random matrix with uniform distribution"""
        return cls(np.random.uniform(low, high, (rows, cols)))
    
    def __str__(self) -> str:
        if self.mode == 'numerical':
            return np.array2string(self.data, precision=6, suppress_small=True)
        return str(self.data)
    
    def __repr__(self) -> str:
        return f"Matrix(shape={self.shape}, mode='{self.mode}')"
    
    def __getitem__(self, key):
        """Get element or slice"""
        return self.data[key]
    
    def __setitem__(self, key, value):
        """Set element or slice"""
        self.data[key] = value
    
    @validate_mode
    def add(self, other: 'Matrix') -> 'Matrix':
        """Add two matrices"""
        self._check_dimensions(other, "addition")
        return Matrix(self.data + other.data, symbolic=(self.mode == 'symbolic'))
    
    @validate_mode
    def subtract(self, other: 'Matrix') -> 'Matrix':
        """Subtract two matrices"""
        self._check_dimensions(other, "subtraction")
        return Matrix(self.data - other.data, symbolic=(self.mode == 'symbolic'))
    
    @validate_mode
    def multiply(self, other: Union['Matrix', 'Vector', NumericType]) -> Union['Matrix', 'Vector']:
        """
        Multiply matrix by another matrix, vector, or scalar.
        
        Args:
            other: Matrix, Vector, or scalar to multiply by
        
        Returns:
            Result of multiplication (Matrix or Vector depending on input)
        """
        if isinstance(other, Vector):
            if self.shape[1] != len(other):
                raise DimensionError(
                    f"Cannot multiply {self.shape} matrix by vector of length {len(other)}"
                )
            if self.mode == 'numerical':
                result = self.data @ other.data
            else:
                result = self.data * other.data
            return Vector(result, symbolic=(self.mode == 'symbolic'))
        
        elif isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise DimensionError(
                    f"Cannot multiply matrices of shapes {self.shape} and {other.shape}"
                )
            if self.mode == 'numerical':
                result = self.data @ other.data
            else:
                result = self.data * other.data
            return Matrix(result, symbolic=(self.mode == 'symbolic'))
        
        else:  # Scalar multiplication
            return Matrix(self.data * other, symbolic=(self.mode == 'symbolic'))
    
    def transpose(self) -> 'Matrix':
        """Return the transpose of the matrix"""
        if self.mode == 'numerical':
            return Matrix(self.data.T, symbolic=False)
        else:
            return Matrix(self.data.T, symbolic=True)
    
    def conjugate(self) -> 'Matrix':
        """Return the complex conjugate"""
        if self.mode == 'numerical':
            return Matrix(np.conj(self.data), symbolic=False)
        else:
            return Matrix(self.data.conjugate(), symbolic=True)
    
    def adjoint(self) -> 'Matrix':
        """Return the conjugate transpose (Hermitian adjoint)"""
        return self.conjugate().transpose()
    
    def det(self) -> Union[float, complex, sp.Expr]:
        """Compute the determinant"""
        if not self.is_square():
            raise DimensionError("Determinant only defined for square matrices")
        
        if self.mode == 'numerical':
            return linalg.det(self.data)
        else:
            return self.data.det()
    
    def trace(self) -> Union[float, complex, sp.Expr]:
        """Compute the trace (sum of diagonal elements)"""
        if not self.is_square():
            raise DimensionError("Trace only defined for square matrices")
        
        if self.mode == 'numerical':
            return np.trace(self.data)
        else:
            return self.data.trace()
    
    def rank(self) -> int:
        """Compute the rank of the matrix"""
        if self.mode == 'numerical':
            return np.linalg.matrix_rank(self.data)
        else:
            return self.data.rank()
    
    def inverse(self) -> 'Matrix':
        """Compute the matrix inverse"""
        if not self.is_square():
            raise DimensionError("Only square matrices can be inverted")
        
        try:
            if self.mode == 'numerical':
                return Matrix(linalg.inv(self.data), symbolic=False)
            else:
                return Matrix(self.data.inv(), symbolic=True)
        except (np.linalg.LinAlgError, ValueError):
            raise SingularMatrixError("Matrix is singular and cannot be inverted")
    
    def pseudoinverse(self) -> 'Matrix':
        """Compute the Moore-Penrose pseudoinverse"""
        if self.mode == 'numerical':
            return Matrix(np.linalg.pinv(self.data), symbolic=False)
        else:
            raise NotImplementedError("Pseudoinverse not available in symbolic mode")
    
    def eigenvalues(self) -> Union[np.ndarray, dict]:
        """
        Compute eigenvalues.
        
        Returns:
            Numerical mode: numpy array of eigenvalues
            Symbolic mode: dict mapping eigenvalue to multiplicity
        """
        if not self.is_square():
            raise DimensionError("Eigenvalues only defined for square matrices")
        
        if self.mode == 'numerical':
            return linalg.eigvals(self.data)
        else:
            return self.data.eigenvals()
    
    def eigenvectors(self) -> Union[Tuple[np.ndarray, np.ndarray], List]:
        """
        Compute eigenvectors.
        
        Returns:
            Numerical mode: (eigenvalues, eigenvectors) tuple
            Symbolic mode: list of (eigenvalue, multiplicity, eigenvectors)
        """
        if not self.is_square():
            raise DimensionError("Eigenvectors only defined for square matrices")
        
        if self.mode == 'numerical':
            return linalg.eig(self.data)
        else:
            return self.data.eigenvects()
    
    def svd(self) -> Tuple['Matrix', Union[np.ndarray, 'Matrix'], 'Matrix']:
        """
        Compute Singular Value Decomposition: A = U * Σ * V†
        
        Returns:
            (U, Σ, V†) where Σ is returned as diagonal values
        """
        if self.mode == 'numerical':
            U, s, Vh = linalg.svd(self.data, full_matrices=False)
            return Matrix(U), s, Matrix(Vh)
        else:
            raise NotImplementedError("SVD not available in symbolic mode")
    
    def qr(self) -> Tuple['Matrix', 'Matrix']:
        """
        Compute QR decomposition: A = Q * R
        
        Returns:
            (Q, R) where Q is orthogonal and R is upper triangular
        """
        if self.mode == 'numerical':
            Q, R = linalg.qr(self.data)
            return Matrix(Q), Matrix(R)
        else:
            Q, R = self.data.QRdecomposition()
            return Matrix(Q, symbolic=True), Matrix(R, symbolic=True)
    
    def lu(self) -> Tuple['Matrix', 'Matrix']:
        """
        Compute LU decomposition: A = L * U
        
        Returns:
            (L, U) where L is lower triangular and U is upper triangular
        """
        if self.mode == 'numerical':
            P, L, U = linalg.lu(self.data)
            # Return L and U (P is permutation matrix)
            return Matrix(L), Matrix(U)
        else:
            L, U, _ = self.data.LUdecomposition()
            return Matrix(L, symbolic=True), Matrix(U, symbolic=True)
    
    def cholesky(self) -> 'Matrix':
        """
        Compute Cholesky decomposition: A = L * L†
        
        Returns:
            L (lower triangular matrix)
        
        Note: Matrix must be positive definite
        """
        if not self.is_square():
            raise DimensionError("Cholesky decomposition only for square matrices")
        
        try:
            if self.mode == 'numerical':
                L = linalg.cholesky(self.data, lower=True)
                return Matrix(L)
            else:
                L = self.data.cholesky()
                return Matrix(L, symbolic=True)
        except (np.linalg.LinAlgError, ValueError):
            raise ValueError("Matrix must be positive definite for Cholesky decomposition")
    
    def solve(self, b: Union[Vector, 'Matrix']) -> Union[Vector, 'Matrix']:
        """
        Solve the linear system Ax = b.
        
        Args:
            b: Right-hand side (Vector or Matrix for multiple RHS)
        
        Returns:
            Solution vector or matrix
        """
        if isinstance(b, Vector):
            if self.shape[0] != len(b):
                raise DimensionError(
                    f"Matrix rows ({self.shape[0]}) must match vector length ({len(b)})"
                )
            if self.mode == 'numerical':
                x = linalg.solve(self.data, b.data)
                return Vector(x, symbolic=False)
            else:
                x = self.data.solve(b.data)
                return Vector(x, symbolic=True)
        
        elif isinstance(b, Matrix):
            if self.shape[0] != b.shape[0]:
                raise DimensionError(
                    f"Matrix rows must match: {self.shape[0]} vs {b.shape[0]}"
                )
            if self.mode == 'numerical':
                x = linalg.solve(self.data, b.data)
                return Matrix(x, symbolic=False)
            else:
                x = self.data.solve(b.data)
                return Matrix(x, symbolic=True)
        else:
            raise TypeError("b must be a Vector or Matrix")
    
    def lstsq(self, b: Vector) -> Vector:
        """
        Compute least-squares solution to Ax = b
        
        Args:
            b: Right-hand side vector
        
        Returns:
            Least-squares solution
        """
        if self.mode == 'numerical':
            x, _, _, _ = linalg.lstsq(self.data, b.data)
            return Vector(x, symbolic=False)
        else:
            raise NotImplementedError("Least squares not available in symbolic mode")
    
    def nullspace(self) -> List[Vector]:
        """
        Compute the nullspace (kernel) of the matrix.
        
        Returns:
            List of basis vectors for the nullspace
        """
        if self.mode == 'numerical':
            # Use SVD to find nullspace
            _, s, Vh = linalg.svd(self.data)
            null_mask = s < 1e-10
            null_space = Vh[null_mask, :]
            return [Vector(v, symbolic=False) for v in null_space]
        else:
            ns = self.data.nullspace()
            return [Vector(v, symbolic=True) for v in ns]
    
    def columnspace(self) -> List[Vector]:
        """
        Compute basis vectors for the column space.
        
        Returns:
            List of basis vectors
        """
        if self.mode == 'numerical':
            Q, _ = linalg.qr(self.data)
            r = self.rank()
            return [Vector(Q[:, i], symbolic=False) for i in range(r)]
        else:
            cs = self.data.columnspace()
            return [Vector(v, symbolic=True) for v in cs]
    
    def rowspace(self) -> List[Vector]:
        """
        Compute basis vectors for the row space.
        
        Returns:
            List of basis vectors
        """
        return self.transpose().columnspace()
    
    def condition_number(self, p: Optional[Union[int, float, str]] = None) -> float:
        """
        Compute the condition number of the matrix.
        
        Args:
            p: Order of the norm (None=2-norm, 1, 2, inf, 'fro')
        """
        if self.mode == 'numerical':
            return np.linalg.cond(self.data, p=p)
        else:
            raise NotImplementedError("Condition number not available in symbolic mode")
    
    def is_square(self) -> bool:
        """Check if matrix is square"""
        return self.shape[0] == self.shape[1]
    
    def is_symmetric(self, tol: float = 1e-10) -> bool:
        """Check if matrix is symmetric"""
        if not self.is_square():
            return False
        if self.mode == 'numerical':
            return np.allclose(self.data, self.data.T, atol=tol)
        else:
            return (self.data - self.data.T).is_zero
    
    def is_hermitian(self, tol: float = 1e-10) -> bool:
        """Check if matrix is Hermitian (self-adjoint)"""
        if not self.is_square():
            return False
        if self.mode == 'numerical':
            return np.allclose(self.data, self.data.T.conj(), atol=tol)
        else:
            return (self.data - self.data.H).is_zero
    
    def is_orthogonal(self, tol: float = 1e-10) -> bool:
        """Check if matrix is orthogonal (Q†Q = I)"""
        if not self.is_square():
            return False
        if self.mode == 'numerical':
            prod = self.data.T @ self.data
            return np.allclose(prod, np.eye(self.shape[0]), atol=tol)
        else:
            prod = self.data.T * self.data
            return (prod - sp.eye(self.shape[0])).is_zero
    
    def is_positive_definite(self) -> bool:
        """Check if matrix is positive definite"""
        if not self.is_square() or not self.is_symmetric():
            return False
        if self.mode == 'numerical':
            try:
                linalg.cholesky(self.data)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return self.data.is_positive_definite
    
    def frobenius_norm(self) -> Union[float, sp.Expr]:
        """Compute Frobenius norm"""
        if self.mode == 'numerical':
            return np.linalg.norm(self.data, 'fro')
        else:
            return sp.sqrt(sum(sp.Abs(x)**2 for x in self.data))
    
    def to_symbolic(self) -> 'Matrix':
        """Convert to symbolic mode"""
        if self.mode == 'symbolic':
            return self
        return Matrix(sp.Matrix(self.data), symbolic=True)
    
    def to_numerical(self) -> 'Matrix':
        """Convert to numerical mode"""
        if self.mode == 'numerical':
            return self
        try:
            rows, cols = self.shape
            data = [[complex(self.data[i, j]) for j in range(cols)] for i in range(rows)]
            return Matrix(data, symbolic=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot convert symbolic matrix to numerical: {e}")
    
    def to_list(self) -> List[List]:
        """Convert to Python nested list"""
        if self.mode == 'numerical':
            return self.data.tolist()
        rows, cols = self.shape
        return [[self.data[i, j] for j in range(cols)] for i in range(rows)]
    
    def _check_dimensions(self, other: 'Matrix', operation: str):
        """Check dimension compatibility"""
        if self.shape != other.shape:
            raise DimensionError(
                f"Cannot perform {operation} on matrices of different shapes: "
                f"{self.shape} vs {other.shape}"
            )


# Utility functions
def tensor_product(v1: Vector, v2: Vector) -> Matrix:
    """Compute tensor (outer) product of two vectors"""
    if v1.mode != v2.mode:
        raise ModeError("Vectors must be in same mode for tensor product")
    
    if v1.mode == 'numerical':
        result = np.outer(v1.data, v2.data)
        return Matrix(result, symbolic=False)
    else:
        result = v1.data * v2.data.T
        return Matrix(result, symbolic=True)


def gram_schmidt(vectors: List[Vector]) -> List[Vector]:
    """
    Apply Gram-Schmidt orthogonalization to a list of vectors.
    
    Args:
        vectors: List of vectors to orthogonalize
    
    Returns:
        List of orthonormal vectors
    """
    if not vectors:
        return []
    
    mode = vectors[0].mode
    if not all(v.mode == mode for v in vectors):
        raise ModeError("All vectors must be in same mode")
    
    orthonormal = []
    for v in vectors:
        # Subtract projections onto previous vectors
        u = v
        for basis in orthonormal:
            u = u.subtract(v.project_onto(basis))
        
        # Normalize
        norm = u.norm()
        if mode == 'numerical' and not np.isclose(norm, 0):
            orthonormal.append(u.normalize())
        elif mode == 'symbolic' and norm != 0:
            orthonormal.append(u.normalize())
    
    return orthonormal


def matrix_power(matrix: Matrix, n: int) -> Matrix:
    """
    Compute matrix raised to integer power n.
    
    Args:
        matrix: Square matrix
        n: Integer exponent (can be negative for inverse powers)
    
    Returns:
        Matrix^n
    """
    if not matrix.is_square():
        raise DimensionError("Matrix must be square for power operation")
    
    if n == 0:
        return Matrix.identity(matrix.shape[0], symbolic=(matrix.mode == 'symbolic'))
    
    if n < 0:
        matrix = matrix.inverse()
        n = -n
    
    if matrix.mode == 'numerical':
        return Matrix(np.linalg.matrix_power(matrix.data, n), symbolic=False)
    else:
        return Matrix(matrix.data ** n, symbolic=True)


def matrix_exp(matrix: Matrix) -> Matrix:
    """
    Compute matrix exponential exp(A).
    
    Args:
        matrix: Square matrix
    
    Returns:
        exp(A)
    """
    if not matrix.is_square():
        raise DimensionError("Matrix must be square for exponential")
    
    if matrix.mode == 'numerical':
        return Matrix(linalg.expm(matrix.data), symbolic=False)
    else:
        return Matrix(matrix.data.exp(), symbolic=True)


def matrix_log(matrix: Matrix) -> Matrix:
    """
    Compute matrix logarithm log(A).
    
    Args:
        matrix: Square matrix
    
    Returns:
        log(A)
    """
    if not matrix.is_square():
        raise DimensionError("Matrix must be square for logarithm")
    
    if matrix.mode == 'numerical':
        return Matrix(linalg.logm(matrix.data), symbolic=False)
    else:
        raise NotImplementedError("Matrix logarithm not available in symbolic mode")


__all__ = [
    'Vector', 'Matrix',
    'VecMatrixError', 'ModeError', 'DimensionError', 'SingularMatrixError',
    'tensor_product', 'gram_schmidt', 'matrix_power', 'matrix_exp', 'matrix_log'
]
