"""
VecMatrix Examples
==================

Comprehensive examples demonstrating various use cases and capabilities.

Run this file to see all examples:
    python examples.py

Author: ML
Version: 2.0.0
"""

import numpy as np
import sympy as sp
from core import (
    Vector, Matrix,
    tensor_product, gram_schmidt, matrix_power, matrix_exp, matrix_log
)


def print_section(title):
    """Print a section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def example_basic_vectors():
    """Basic vector operations"""
    print_section("Basic Vector Operations")
    
    # Create vectors
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print()
    
    # Operations
    print(f"v1 + v2 = {v1.add(v2)}")
    print(f"v1 - v2 = {v1.subtract(v2)}")
    print(f"v1 · v2 = {v1.dot(v2)}")
    print(f"||v1|| = {v1.norm():.4f}")
    print(f"v1 / ||v1|| = {v1.normalize()}")
    print()
    
    # 3D cross product
    v3 = Vector([1, 0, 0])
    v4 = Vector([0, 1, 0])
    print(f"[1,0,0] × [0,1,0] = {v3.cross(v4)}")


def example_basic_matrices():
    """Basic matrix operations"""
    print_section("Basic Matrix Operations")
    
    # Create matrices
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    
    print(f"A =\n{A}\n")
    print(f"B =\n{B}\n")
    
    # Operations
    print(f"A + B =\n{A.add(B)}\n")
    print(f"A × B =\n{A.multiply(B)}\n")
    print(f"A^T =\n{A.transpose()}\n")
    print(f"det(A) = {A.det()}")
    print(f"trace(A) = {A.trace()}")
    print(f"rank(A) = {A.rank()}")


def example_solving_systems():
    """Solving linear systems"""
    print_section("Solving Linear Systems")
    
    # Standard square system: Ax = b
    A = Matrix([[2, 1], [1, 3]])
    b = Vector([5, 8])
    
    print(f"Solve Ax = b:")
    print(f"A =\n{A}\n")
    print(f"b = {b}\n")
    
    x = A.solve(b)
    print(f"Solution: x = {x}")
    
    # Verify
    result = A.multiply(x)
    print(f"Verification: Ax = {result}")
    print()
    
    # Overdetermined system (least squares)
    print("Least Squares Solution (overdetermined system):")
    A_over = Matrix([[1, 1], [1, 2], [1, 3]])
    b_over = Vector([1, 2, 2])
    
    print(f"A =\n{A_over}\n")
    print(f"b = {b_over}\n")
    
    x_ls = A_over.lstsq(b_over)
    print(f"Least squares solution: x = {x_ls}")


def example_eigenanalysis():
    """Eigenvalue and eigenvector computation"""
    print_section("Eigenanalysis")
    
    # Symmetric matrix
    A = Matrix([[1, 2], [2, 1]])
    print(f"Matrix A =\n{A}\n")
    
    # Eigenvalues
    eigenvals = A.eigenvalues()
    print(f"Eigenvalues: {eigenvals}")
    print()
    
    # Eigenvectors
    evals, evecs = A.eigenvectors()
    print("Eigenvalue-Eigenvector pairs:")
    for i in range(len(evals)):
        eval_i = evals[i]
        evec_i = Vector(evecs[:, i])
        print(f"λ = {eval_i:.4f}, v = {evec_i}")
        
        # Verify Av = λv
        Av = A.multiply(evec_i)
        lambda_v = evec_i.scale(eval_i)
        print(f"  Av = {Av}")
        print(f"  λv = {lambda_v}")
        print()


def example_decompositions():
    """Matrix decompositions"""
    print_section("Matrix Decompositions")
    
    # Create test matrix
    A = Matrix([[1, 2], [3, 4], [5, 6]])
    print(f"Matrix A (3×2) =\n{A}\n")
    
    # SVD
    print("Singular Value Decomposition: A = UΣV†")
    U, sigma, Vh = A.svd()
    print(f"U =\n{U}\n")
    print(f"Σ (singular values) = {sigma}")
    print(f"V† =\n{Vh}\n")
    
    # QR
    print("QR Decomposition: A = QR")
    Q, R = A.qr()
    print(f"Q (orthogonal) =\n{Q}\n")
    print(f"R (upper triangular) =\n{R}\n")
    
    # Verify orthogonality of Q
    QtQ = Q.transpose().multiply(Q)
    print(f"Q†Q (should be I) =\n{QtQ}\n")
    
    # LU for square matrix
    B = Matrix([[2, 1], [4, 3]])
    print(f"Square matrix B =\n{B}\n")
    print("LU Decomposition: B = LU")
    L, U = B.lu()
    print(f"L (lower triangular) =\n{L}\n")
    print(f"U (upper triangular) =\n{U}\n")
    
    # Cholesky for positive definite
    C = Matrix([[4, 2], [2, 3]])
    print(f"Positive definite C =\n{C}\n")
    print("Cholesky Decomposition: C = LL†")
    L_chol = C.cholesky()
    print(f"L =\n{L_chol}\n")
    
    # Verify
    reconstructed = L_chol.multiply(L_chol.transpose())
    print(f"LL† (should equal C) =\n{reconstructed}")


def example_subspaces():
    """Vector subspaces"""
    print_section("Vector Subspaces")
    
    # Matrix with rank < min(m,n)
    A = Matrix([[1, 2, 3], [2, 4, 6], [1, 1, 1]])
    print(f"Matrix A =\n{A}\n")
    print(f"Rank = {A.rank()}\n")
    
    # Nullspace
    print("Nullspace (vectors v where Av = 0):")
    nullspace = A.nullspace()
    for i, v in enumerate(nullspace):
        print(f"Basis vector {i+1}: {v}")
        # Verify Av = 0
        Av = A.multiply(v)
        print(f"  Av = {Av} (should be ~0)\n")
    
    # Column space
    print("Column space:")
    colspace = A.columnspace()
    for i, v in enumerate(colspace):
        print(f"Basis vector {i+1}: {v}")
    print()
    
    # Row space
    print("Row space:")
    rowspace = A.rowspace()
    for i, v in enumerate(rowspace):
        print(f"Basis vector {i+1}: {v}")


def example_gram_schmidt():
    """Gram-Schmidt orthogonalization"""
    print_section("Gram-Schmidt Orthogonalization")
    
    # Non-orthogonal vectors
    v1 = Vector([1, 1, 0])
    v2 = Vector([1, 0, 1])
    v3 = Vector([0, 1, 1])
    
    print("Original vectors:")
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v3 = {v3}")
    print()
    
    # Check they're not orthogonal
    print("Dot products (non-zero means not orthogonal):")
    print(f"v1·v2 = {v1.dot(v2)}")
    print(f"v1·v3 = {v1.dot(v3)}")
    print(f"v2·v3 = {v2.dot(v3)}")
    print()
    
    # Apply Gram-Schmidt
    orthonormal = gram_schmidt([v1, v2, v3])
    
    print("Orthonormal basis:")
    for i, v in enumerate(orthonormal):
        print(f"u{i+1} = {v}")
        print(f"  ||u{i+1}|| = {v.norm():.6f}")
    print()
    
    # Verify orthogonality
    print("Verification (all dot products should be ~0):")
    for i, ui in enumerate(orthonormal):
        for j, uj in enumerate(orthonormal):
            if i < j:
                print(f"u{i+1}·u{j+1} = {ui.dot(uj):.10f}")


def example_matrix_functions():
    """Matrix functions"""
    print_section("Matrix Functions")
    
    # Matrix power
    A = Matrix([[1, 1], [0, 1]])
    print(f"Matrix A =\n{A}\n")
    
    print("Matrix powers:")
    for n in [2, 3, -1]:
        A_n = matrix_power(A, n)
        print(f"A^{n} =\n{A_n}\n")
    
    # Matrix exponential
    B = Matrix([[0, 1], [0, 0]])
    print(f"Matrix B =\n{B}\n")
    
    exp_B = matrix_exp(B)
    print(f"exp(B) =\n{exp_B}\n")
    
    # Matrix logarithm
    C = Matrix([[1, 1], [0, 1]])
    print(f"Matrix C =\n{C}\n")
    
    log_C = matrix_log(C)
    print(f"log(C) =\n{log_C}\n")
    
    # Verify exp(log(C)) = C
    reconstructed = matrix_exp(log_C)
    print(f"exp(log(C)) (should equal C) =\n{reconstructed}")


def example_symbolic():
    """Symbolic computation"""
    print_section("Symbolic Computation")
    
    # Symbolic vectors
    print("Symbolic vectors:")
    x, y, z = sp.symbols('x y z')
    v = Vector([x, y, z], symbolic=True)
    print(f"v = {v}")
    print(f"||v|| = {v.norm()}")
    print()
    
    # Symbolic dot product
    a, b, c = sp.symbols('a b c')
    v1 = Vector([x, y, z], symbolic=True)
    v2 = Vector([a, b, c], symbolic=True)
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1·v2 = {v1.dot(v2)}")
    print()
    
    # Symbolic matrix
    print("Symbolic matrices:")
    alpha, beta, gamma, delta = sp.symbols('alpha beta gamma delta')
    M = Matrix([[alpha, beta], [gamma, delta]], symbolic=True)
    print(f"M =\n{M}\n")
    print(f"det(M) = {M.det()}")
    print(f"trace(M) = {M.trace()}")
    print()
    
    # Symbolic inverse
    print(f"M^(-1) =\n{M.inverse()}")


def example_complex_numbers():
    """Complex number support"""
    print_section("Complex Numbers")
    
    # Complex vectors
    v = Vector([1+2j, 3+4j])
    print(f"Complex vector: v = {v}")
    print(f"||v|| = {v.norm():.4f}")
    print()
    
    # Complex matrix
    A = Matrix([[1+1j, 2], [3, 4-1j]])
    print(f"Complex matrix A =\n{A}\n")
    
    # Conjugate
    A_conj = A.conjugate()
    print(f"Complex conjugate:\n{A_conj}\n")
    
    # Adjoint (conjugate transpose)
    A_adj = A.adjoint()
    print(f"Adjoint (Hermitian transpose):\n{A_adj}\n")
    
    # Hermitian matrix
    H = Matrix([[1, 1+1j], [1-1j, 2]])
    print(f"Hermitian matrix H =\n{H}\n")
    print(f"Is Hermitian? {H.is_hermitian()}")
    print(f"H† =\n{H.adjoint()}\n")
    print(f"H† == H? {np.allclose(H.data, H.adjoint().data)}")


def example_tensor_product():
    """Tensor (outer) products"""
    print_section("Tensor Products")
    
    v1 = Vector([1, 2])
    v2 = Vector([3, 4, 5])
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print()
    
    M = tensor_product(v1, v2)
    print(f"v1 ⊗ v2 =\n{M}")


def example_applications():
    """Real-world applications"""
    print_section("Real-World Applications")
    
    print("Example 1: Least Squares Regression")
    print("-" * 50)
    # Fit line y = mx + b to data points
    x_data = [1, 2, 3, 4, 5]
    y_data = [2.1, 3.9, 6.2, 7.8, 10.1]
    
    # Build design matrix [1, x]
    A = Matrix([[1, x] for x in x_data])
    b = Vector(y_data)
    
    print(f"Data points: {list(zip(x_data, y_data))}")
    print(f"Design matrix A =\n{A}\n")
    print(f"Observations b = {b}\n")
    
    # Solve least squares
    params = A.lstsq(b)
    intercept, slope = params[0], params[1]
    print(f"Best fit line: y = {slope:.4f}x + {intercept:.4f}")
    print()
    
    print("Example 2: Computing Projection Matrix")
    print("-" * 50)
    # Project onto column space of A
    A = Matrix([[1, 0], [0, 1], [1, 1]])
    print(f"Matrix A =\n{A}\n")
    
    # Projection matrix P = A(A^T A)^-1 A^T
    At = A.transpose()
    AtA = At.multiply(A)
    AtA_inv = AtA.inverse()
    P = A.multiply(AtA_inv).multiply(At)
    
    print(f"Projection matrix P =\n{P}\n")
    
    # Verify P^2 = P (idempotent)
    P2 = P.multiply(P)
    print(f"P² (should equal P) =\n{P2}\n")
    print(f"Is P² ≈ P? {np.allclose(P.data, P2.data)}")


def run_all_examples():
    """Run all examples"""
    example_basic_vectors()
    example_basic_matrices()
    example_solving_systems()
    example_eigenanalysis()
    example_decompositions()
    example_subspaces()
    example_gram_schmidt()
    example_matrix_functions()
    example_symbolic()
    example_complex_numbers()
    example_tensor_product()
    example_applications()
    
    print_section("Examples Complete!")
    print("For more information, see README.md")
    print("Run tests with: python test_vecmatrix.py")
    print("Start API server with: python api_server.py")
    print()


if __name__ == '__main__':
    run_all_examples()
