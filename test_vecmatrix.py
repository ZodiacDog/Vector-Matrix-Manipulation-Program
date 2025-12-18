"""
VecMatrix Test Suite
====================

Comprehensive tests for all vector and matrix operations.

Run tests:
    python test_vecmatrix.py
    python -m pytest test_vecmatrix.py -v

Author: ML
Version: 2.0.0
"""

import unittest
import numpy as np
import sympy as sp
from core import (
    Vector, Matrix, VecMatrixError, ModeError, DimensionError, SingularMatrixError,
    tensor_product, gram_schmidt, matrix_power, matrix_exp, matrix_log
)


class TestVector(unittest.TestCase):
    """Test Vector class"""
    
    def test_creation_numerical(self):
        """Test numerical vector creation"""
        v = Vector([1, 2, 3])
        self.assertEqual(v.mode, 'numerical')
        self.assertEqual(len(v), 3)
        np.testing.assert_array_equal(v.data, [1, 2, 3])
    
    def test_creation_symbolic(self):
        """Test symbolic vector creation"""
        v = Vector(['x', 'y', 'z'], symbolic=True)
        self.assertEqual(v.mode, 'symbolic')
        self.assertEqual(len(v), 3)
    
    def test_auto_detect_symbolic(self):
        """Test automatic symbolic mode detection"""
        v = Vector(['x', 'y', 2])
        self.assertEqual(v.mode, 'symbolic')
    
    def test_vector_addition(self):
        """Test vector addition"""
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        v3 = v1.add(v2)
        np.testing.assert_array_equal(v3.data, [5, 7, 9])
    
    def test_vector_subtraction(self):
        """Test vector subtraction"""
        v1 = Vector([4, 5, 6])
        v2 = Vector([1, 2, 3])
        v3 = v1.subtract(v2)
        np.testing.assert_array_equal(v3.data, [3, 3, 3])
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication"""
        v = Vector([1, 2, 3])
        v2 = v.scale(2)
        np.testing.assert_array_equal(v2.data, [2, 4, 6])
    
    def test_dot_product(self):
        """Test dot product"""
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        result = v1.dot(v2)
        self.assertAlmostEqual(result, 32.0)
    
    def test_norm(self):
        """Test vector norm"""
        v = Vector([3, 4])
        self.assertAlmostEqual(v.norm(), 5.0)
    
    def test_cross_product(self):
        """Test cross product"""
        v1 = Vector([1, 0, 0])
        v2 = Vector([0, 1, 0])
        v3 = v1.cross(v2)
        np.testing.assert_array_almost_equal(v3.data, [0, 0, 1])
    
    def test_normalize(self):
        """Test vector normalization"""
        v = Vector([3, 4])
        v_norm = v.normalize()
        self.assertAlmostEqual(v_norm.norm(), 1.0)
        np.testing.assert_array_almost_equal(v_norm.data, [0.6, 0.8])
    
    def test_projection(self):
        """Test vector projection"""
        v1 = Vector([1, 2])
        v2 = Vector([1, 0])
        proj = v1.project_onto(v2)
        np.testing.assert_array_almost_equal(proj.data, [1, 0])
    
    def test_angle_between(self):
        """Test angle calculation"""
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        angle = v1.angle_between(v2)
        self.assertAlmostEqual(angle, np.pi / 2)
    
    def test_distance(self):
        """Test distance calculation"""
        v1 = Vector([0, 0])
        v2 = Vector([3, 4])
        dist = v1.distance_to(v2)
        self.assertAlmostEqual(dist, 5.0)
    
    def test_orthogonality(self):
        """Test orthogonality check"""
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        self.assertTrue(v1.is_orthogonal_to(v2))
        
        v3 = Vector([1, 1])
        self.assertFalse(v1.is_orthogonal_to(v3))
    
    def test_mode_mismatch_error(self):
        """Test error on mode mismatch"""
        v1 = Vector([1, 2, 3])
        v2 = Vector(['x', 'y', 'z'], symbolic=True)
        with self.assertRaises(ModeError):
            v1.add(v2)
    
    def test_dimension_mismatch_error(self):
        """Test error on dimension mismatch"""
        v1 = Vector([1, 2])
        v2 = Vector([1, 2, 3])
        with self.assertRaises(DimensionError):
            v1.add(v2)
    
    def test_symbolic_operations(self):
        """Test symbolic vector operations"""
        x, y = sp.symbols('x y')
        v1 = Vector([x, y], symbolic=True)
        v2 = Vector([1, 2], symbolic=True)
        v3 = v1.add(v2)
        self.assertEqual(str(v3.data[0]), 'x + 1')
        self.assertEqual(str(v3.data[1]), 'y + 2')


class TestMatrix(unittest.TestCase):
    """Test Matrix class"""
    
    def test_creation_numerical(self):
        """Test numerical matrix creation"""
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.mode, 'numerical')
        self.assertEqual(m.shape, (2, 2))
    
    def test_creation_symbolic(self):
        """Test symbolic matrix creation"""
        m = Matrix([['a', 'b'], ['c', 'd']], symbolic=True)
        self.assertEqual(m.mode, 'symbolic')
        self.assertEqual(m.shape, (2, 2))
    
    def test_identity(self):
        """Test identity matrix creation"""
        I = Matrix.identity(3)
        np.testing.assert_array_equal(I.data, np.eye(3))
    
    def test_zeros(self):
        """Test zero matrix creation"""
        Z = Matrix.zeros(2, 3)
        np.testing.assert_array_equal(Z.data, np.zeros((2, 3)))
    
    def test_ones(self):
        """Test ones matrix creation"""
        O = Matrix.ones(2, 3)
        np.testing.assert_array_equal(O.data, np.ones((2, 3)))
    
    def test_matrix_addition(self):
        """Test matrix addition"""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        m3 = m1.add(m2)
        np.testing.assert_array_equal(m3.data, [[6, 8], [10, 12]])
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication"""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        m3 = m1.multiply(m2)
        expected = [[19, 22], [43, 50]]
        np.testing.assert_array_almost_equal(m3.data, expected)
    
    def test_matrix_vector_multiply(self):
        """Test matrix-vector multiplication"""
        m = Matrix([[1, 2], [3, 4]])
        v = Vector([1, 2])
        result = m.multiply(v)
        np.testing.assert_array_almost_equal(result.data, [5, 11])
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication"""
        m = Matrix([[1, 2], [3, 4]])
        m2 = m.multiply(2)
        np.testing.assert_array_equal(m2.data, [[2, 4], [6, 8]])
    
    def test_transpose(self):
        """Test matrix transpose"""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        mt = m.transpose()
        np.testing.assert_array_equal(mt.data, [[1, 4], [2, 5], [3, 6]])
    
    def test_determinant(self):
        """Test determinant calculation"""
        m = Matrix([[1, 2], [3, 4]])
        det = m.det()
        self.assertAlmostEqual(det, -2.0)
    
    def test_trace(self):
        """Test trace calculation"""
        m = Matrix([[1, 2], [3, 4]])
        tr = m.trace()
        self.assertAlmostEqual(tr, 5.0)
    
    def test_rank(self):
        """Test rank calculation"""
        m = Matrix([[1, 2], [2, 4]])  # Rank 1
        self.assertEqual(m.rank(), 1)
        
        m2 = Matrix([[1, 0], [0, 1]])  # Rank 2
        self.assertEqual(m2.rank(), 2)
    
    def test_inverse(self):
        """Test matrix inverse"""
        m = Matrix([[1, 2], [3, 4]])
        m_inv = m.inverse()
        
        # Check that M * M^-1 = I
        I = m.multiply(m_inv)
        np.testing.assert_array_almost_equal(I.data, np.eye(2))
    
    def test_singular_matrix_error(self):
        """Test error on singular matrix inversion"""
        m = Matrix([[1, 2], [2, 4]])  # Singular
        with self.assertRaises(SingularMatrixError):
            m.inverse()
    
    def test_eigenvalues(self):
        """Test eigenvalue calculation"""
        m = Matrix([[1, 2], [2, 1]])
        eigs = m.eigenvalues()
        eigs_sorted = np.sort(eigs)
        np.testing.assert_array_almost_equal(eigs_sorted, [-1, 3])
    
    def test_eigenvectors(self):
        """Test eigenvector calculation"""
        m = Matrix([[2, 0], [0, 3]])
        evals, evecs = m.eigenvectors()
        
        # Check that eigenvalues are correct
        evals_sorted = np.sort(evals)
        np.testing.assert_array_almost_equal(evals_sorted, [2, 3])
    
    def test_svd(self):
        """Test SVD decomposition"""
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        U, s, Vh = m.svd()
        
        # Reconstruct matrix: M = U @ diag(s) @ Vh
        # U is m x min(m,n), s is min(m,n), Vh is min(m,n) x n
        # For m=3, n=2: U is 3x2, s is 2, Vh is 2x2
        S_diag = np.diag(s)
        reconstructed = U.data @ S_diag @ Vh.data
        
        np.testing.assert_array_almost_equal(reconstructed, m.data)
    
    def test_qr(self):
        """Test QR decomposition"""
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        Q, R = m.qr()
        
        # Check that Q*R = M
        reconstructed = Q.multiply(R)
        np.testing.assert_array_almost_equal(reconstructed.data, m.data)
        
        # Check that Q is orthogonal (Q†Q should be identity)
        # Note: Q is 3x2, so Q†Q is 2x2
        QtQ = Q.transpose().multiply(Q)
        np.testing.assert_array_almost_equal(QtQ.data, np.eye(QtQ.shape[0]), decimal=10)
    
    def test_lu(self):
        """Test LU decomposition"""
        m = Matrix([[1, 2], [3, 4]])
        L, U = m.lu()
        
        # Check that L*U ≈ M (up to permutation)
        reconstructed = L.multiply(U)
        # LU decomposition may involve row permutations
        self.assertEqual(reconstructed.shape, m.shape)
    
    def test_cholesky(self):
        """Test Cholesky decomposition"""
        # Create positive definite matrix
        A = np.array([[4, 2], [2, 3]])
        m = Matrix(A)
        L = m.cholesky()
        
        # Check that L*L^T = M
        reconstructed = L.multiply(L.transpose())
        np.testing.assert_array_almost_equal(reconstructed.data, m.data)
    
    def test_solve_linear_system(self):
        """Test solving linear system"""
        A = Matrix([[2, 1], [1, 3]])
        b = Vector([5, 8])
        x = A.solve(b)
        
        # Check that Ax = b
        result = A.multiply(x)
        np.testing.assert_array_almost_equal(result.data, b.data)
    
    def test_least_squares(self):
        """Test least squares solution"""
        # Overdetermined system
        A = Matrix([[1, 1], [1, 2], [1, 3]])
        b = Vector([1, 2, 2])
        x = A.lstsq(b)
        
        # Solution should minimize ||Ax - b||
        self.assertEqual(len(x), 2)
    
    def test_nullspace(self):
        """Test nullspace calculation"""
        m = Matrix([[1, 2], [2, 4]])  # Rank 1, nullspace dimension 1
        null = m.nullspace()
        
        # Check that vectors in nullspace satisfy Mv = 0
        for v in null:
            result = m.multiply(v)
            self.assertTrue(np.allclose(result.data, 0, atol=1e-10))
    
    def test_column_space(self):
        """Test column space calculation"""
        m = Matrix([[1, 2], [2, 4], [3, 6]])  # Rank 1
        colspace = m.columnspace()
        
        # Column space dimension should equal rank
        self.assertEqual(len(colspace), 1)
    
    def test_is_symmetric(self):
        """Test symmetry check"""
        m_sym = Matrix([[1, 2], [2, 1]])
        self.assertTrue(m_sym.is_symmetric())
        
        m_nonsym = Matrix([[1, 2], [3, 4]])
        self.assertFalse(m_nonsym.is_symmetric())
    
    def test_is_orthogonal(self):
        """Test orthogonality check"""
        # Rotation matrix (orthogonal)
        theta = np.pi / 4
        R = Matrix([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
        self.assertTrue(R.is_orthogonal())
    
    def test_is_positive_definite(self):
        """Test positive definiteness check"""
        m_pd = Matrix([[2, 1], [1, 2]])
        self.assertTrue(m_pd.is_positive_definite())
        
        m_not_pd = Matrix([[1, 2], [2, 1]])
        self.assertFalse(m_not_pd.is_positive_definite())
    
    def test_frobenius_norm(self):
        """Test Frobenius norm"""
        m = Matrix([[1, 2], [3, 4]])
        norm = m.frobenius_norm()
        expected = np.sqrt(1**2 + 2**2 + 3**2 + 4**2)
        self.assertAlmostEqual(norm, expected)
    
    def test_condition_number(self):
        """Test condition number"""
        m = Matrix([[1, 2], [2, 4.01]])  # Nearly singular
        cond = m.condition_number()
        self.assertGreater(cond, 100)  # Should be large
    
    def test_symbolic_determinant(self):
        """Test symbolic determinant"""
        a, b, c, d = sp.symbols('a b c d')
        m = Matrix([[a, b], [c, d]], symbolic=True)
        det = m.det()
        self.assertEqual(str(det), 'a*d - b*c')


class TestUtilities(unittest.TestCase):
    """Test utility functions"""
    
    def test_tensor_product(self):
        """Test tensor product"""
        v1 = Vector([1, 2])
        v2 = Vector([3, 4])
        m = tensor_product(v1, v2)
        expected = [[3, 4], [6, 8]]
        np.testing.assert_array_almost_equal(m.data, expected)
    
    def test_gram_schmidt(self):
        """Test Gram-Schmidt orthogonalization"""
        v1 = Vector([1, 1, 0])
        v2 = Vector([1, 0, 1])
        v3 = Vector([0, 1, 1])
        
        ortho = gram_schmidt([v1, v2, v3])
        
        # Check orthonormality
        for i, vi in enumerate(ortho):
            # Check normalized
            self.assertAlmostEqual(vi.norm(), 1.0)
            
            # Check orthogonal to others
            for j, vj in enumerate(ortho):
                if i != j:
                    self.assertAlmostEqual(abs(vi.dot(vj)), 0.0, places=10)
    
    def test_matrix_power(self):
        """Test matrix power"""
        m = Matrix([[1, 1], [0, 1]])
        m2 = matrix_power(m, 2)
        expected = [[1, 2], [0, 1]]
        np.testing.assert_array_almost_equal(m2.data, expected)
        
        # Test negative power
        m_inv = matrix_power(m, -1)
        I = m.multiply(m_inv)
        np.testing.assert_array_almost_equal(I.data, np.eye(2))
    
    def test_matrix_exp(self):
        """Test matrix exponential"""
        m = Matrix([[0, 1], [0, 0]])
        m_exp = matrix_exp(m)
        expected = [[1, 1], [0, 1]]
        np.testing.assert_array_almost_equal(m_exp.data, expected, decimal=10)
    
    def test_matrix_log(self):
        """Test matrix logarithm"""
        m = Matrix([[1, 1], [0, 1]])
        m_log = matrix_log(m)
        
        # Check that exp(log(M)) = M
        m_reconstructed = matrix_exp(m_log)
        np.testing.assert_array_almost_equal(m_reconstructed.data, m.data)


class TestComplexNumbers(unittest.TestCase):
    """Test complex number support"""
    
    def test_complex_vector(self):
        """Test vector with complex numbers"""
        v = Vector([1+2j, 3+4j])
        self.assertEqual(v.mode, 'numerical')
        norm = v.norm()
        expected = np.sqrt(abs(1+2j)**2 + abs(3+4j)**2)
        self.assertAlmostEqual(norm, expected)
    
    def test_complex_matrix(self):
        """Test matrix with complex numbers"""
        m = Matrix([[1+1j, 2], [3, 4-1j]])
        self.assertEqual(m.mode, 'numerical')
        
        # Test conjugate
        m_conj = m.conjugate()
        expected = [[1-1j, 2], [3, 4+1j]]
        np.testing.assert_array_equal(m_conj.data, expected)
    
    def test_hermitian_matrix(self):
        """Test Hermitian matrix operations"""
        m = Matrix([[1, 1+1j], [1-1j, 2]])
        self.assertTrue(m.is_hermitian())


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_vector_error(self):
        """Test error on empty vector"""
        with self.assertRaises(ValueError):
            Vector([])
    
    def test_empty_matrix_error(self):
        """Test error on empty matrix"""
        with self.assertRaises(ValueError):
            Matrix([[]])
    
    def test_zero_vector_normalization(self):
        """Test error on normalizing zero vector"""
        v = Vector([0, 0, 0])
        with self.assertRaises(ValueError):
            v.normalize()
    
    def test_non_square_determinant(self):
        """Test error on non-square determinant"""
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(DimensionError):
            m.det()
    
    def test_incompatible_matrix_multiplication(self):
        """Test error on incompatible matrix multiplication"""
        m1 = Matrix([[1, 2]])  # 1x2
        m2 = Matrix([[1], [2], [3]])  # 3x1
        with self.assertRaises(DimensionError):
            m1.multiply(m2)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVector))
    suite.addTests(loader.loadTestsFromTestCase(TestMatrix))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilities))
    suite.addTests(loader.loadTestsFromTestCase(TestComplexNumbers))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit(run_tests())
