# VecMatrix - Advanced Vector & Matrix Library

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/yourusername/vecmatrix)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

**The most comprehensive vector and matrix manipulation library**, featuring dual numerical/symbolic computation modes, extensive linear algebra operations, and multiple interfaces (CLI, API, Library).

## 🚀 Features

### Core Capabilities
- **Dual Mode Operation**: Seamlessly switch between numerical (NumPy) and symbolic (SymPy) computation
- **Comprehensive Operations**: 60+ vector and matrix operations
- **Type Safety**: Full type hints and validation
- **Performance Optimized**: Efficient algorithms with scipy integration
- **Thread Safe**: Production-ready concurrent operation support
- **Complex Numbers**: Full support for complex arithmetic
- **Multiple Interfaces**: Use as library, CLI tool, or REST API

### Advanced Operations
- **Decompositions**: SVD, QR, LU, Cholesky
- **Eigenanalysis**: Eigenvalues, eigenvectors, matrix functions
- **Solving Systems**: Direct solve, least squares, pseudoinverse
- **Spaces**: Nullspace, column space, row space
- **Matrix Functions**: Power, exponential, logarithm
- **Utilities**: Gram-Schmidt, tensor products, condition numbers

## 📦 Installation

```bash
# Basic installation
pip install -r requirements.txt

# Development installation
pip install -e .

# Or install directly
python setup.py install
```

## 🎯 Quick Start

### As a Library

```python
from core import Vector, Matrix

# Create vectors
v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])

# Operations
dot_product = v1.dot(v2)  # 32.0
cross = v1.cross(v2)       # Vector([-3, 6, -3])
norm = v1.norm()           # 3.7416...

# Create matrices
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

# Matrix operations
C = A.multiply(B)          # Matrix multiplication
det = A.det()              # Determinant: -2.0
inv = A.inverse()          # Inverse matrix
eigs = A.eigenvalues()     # [-0.372, 5.372]

# Solve linear system Ax = b
b = Vector([5, 11])
x = A.solve(b)             # Solution: [1, 2]
```

### Symbolic Mode

```python
from core import Vector, Matrix
import sympy as sp

# Symbolic vectors
x, y, z = sp.symbols('x y z')
v = Vector([x, y, z], symbolic=True)
print(v.norm())  # sqrt(x**2 + y**2 + z**2)

# Symbolic matrices
a, b, c, d = sp.symbols('a b c d')
M = Matrix([[a, b], [c, d]], symbolic=True)
print(M.det())  # a*d - b*c
```

### Command Line Interface

```bash
# Vector operations
vecmatrix vector dot '[1,2,3]' '[4,5,6]'
vecmatrix vector cross '[1,0,0]' '[0,1,0]'
vecmatrix vector norm '[3,4]'

# Matrix operations
vecmatrix matrix multiply '[[1,2],[3,4]]' '[[5,6],[7,8]]'
vecmatrix matrix inverse '[[1,2],[3,4]]'
vecmatrix matrix eigenvalues '[[1,2],[2,1]]'

# Symbolic mode
vecmatrix vector dot '[x,y,z]' '[a,b,c]' --symbolic
vecmatrix matrix det '[[a,b],[c,d]]' --symbolic

# Output formats
vecmatrix matrix inverse '[[1,2],[3,4]]' --format json
vecmatrix matrix qr '[[1,2],[3,4],[5,6]]' --format latex
```

### REST API

```bash
# Start the server
python api_server.py --host localhost --port 8000

# Make requests
curl -X POST http://localhost:8000/vector/dot \
  -H "Content-Type: application/json" \
  -d '{"args": [[1,2,3], [4,5,6]]}'

curl -X POST http://localhost:8000/matrix/inverse \
  -H "Content-Type: application/json" \
  -d '{"args": [[[1,2],[3,4]]]}'

# List operations
curl http://localhost:8000/operations
```

## 📚 Documentation

### Vector Operations

| Operation | Description | Example |
|-----------|-------------|---------|
| `add(v)` | Vector addition | `v1.add(v2)` |
| `subtract(v)` | Vector subtraction | `v1.subtract(v2)` |
| `scale(s)` | Scalar multiplication | `v.scale(2)` |
| `dot(v)` | Dot product | `v1.dot(v2)` |
| `cross(v)` | Cross product (3D) | `v1.cross(v2)` |
| `norm(ord)` | Vector norm | `v.norm()` |
| `normalize()` | Unit vector | `v.normalize()` |
| `project_onto(v)` | Projection | `v1.project_onto(v2)` |
| `angle_between(v)` | Angle (radians) | `v1.angle_between(v2)` |
| `distance_to(v)` | Euclidean distance | `v1.distance_to(v2)` |
| `is_orthogonal_to(v)` | Orthogonality test | `v1.is_orthogonal_to(v2)` |

### Matrix Operations

| Operation | Description | Example |
|-----------|-------------|---------|
| `add(M)` | Matrix addition | `A.add(B)` |
| `subtract(M)` | Matrix subtraction | `A.subtract(B)` |
| `multiply(M/v/s)` | Multiplication | `A.multiply(B)` |
| `transpose()` | Transpose | `A.transpose()` |
| `conjugate()` | Complex conjugate | `A.conjugate()` |
| `adjoint()` | Hermitian adjoint | `A.adjoint()` |
| `det()` | Determinant | `A.det()` |
| `trace()` | Trace | `A.trace()` |
| `rank()` | Matrix rank | `A.rank()` |
| `inverse()` | Inverse | `A.inverse()` |
| `pseudoinverse()` | Moore-Penrose | `A.pseudoinverse()` |
| `eigenvalues()` | Eigenvalues | `A.eigenvalues()` |
| `eigenvectors()` | Eigenvectors | `A.eigenvectors()` |
| `svd()` | SVD: U, Σ, V† | `A.svd()` |
| `qr()` | QR decomposition | `A.qr()` |
| `lu()` | LU decomposition | `A.lu()` |
| `cholesky()` | Cholesky | `A.cholesky()` |
| `solve(b)` | Solve Ax=b | `A.solve(b)` |
| `lstsq(b)` | Least squares | `A.lstsq(b)` |
| `nullspace()` | Null space basis | `A.nullspace()` |
| `columnspace()` | Column space basis | `A.columnspace()` |
| `rowspace()` | Row space basis | `A.rowspace()` |

### Matrix Properties

| Method | Description |
|--------|-------------|
| `is_square()` | Check if square |
| `is_symmetric()` | Check if symmetric |
| `is_hermitian()` | Check if Hermitian |
| `is_orthogonal()` | Check if orthogonal |
| `is_positive_definite()` | Check if positive definite |
| `condition_number()` | Condition number |
| `frobenius_norm()` | Frobenius norm |

### Utility Functions

```python
from core import tensor_product, gram_schmidt, matrix_power, matrix_exp, matrix_log

# Tensor (outer) product
M = tensor_product(v1, v2)

# Gram-Schmidt orthogonalization
orthonormal_vectors = gram_schmidt([v1, v2, v3])

# Matrix power
M_squared = matrix_power(M, 2)
M_inv = matrix_power(M, -1)

# Matrix exponential
exp_M = matrix_exp(M)

# Matrix logarithm
log_M = matrix_log(M)
```

### Factory Methods

```python
# Identity matrix
I = Matrix.identity(3)

# Zero matrix
Z = Matrix.zeros(3, 4)

# Ones matrix
O = Matrix.ones(2, 2)

# Random matrix
R = Matrix.random(3, 3, low=-1, high=1)
```

## 🔧 Advanced Examples

### Solving Linear Systems

```python
# Standard solve
A = Matrix([[2, 1], [1, 3]])
b = Vector([5, 8])
x = A.solve(b)  # Exact solution

# Overdetermined system (least squares)
A = Matrix([[1, 1], [1, 2], [1, 3]])
b = Vector([1, 2, 2])
x = A.lstsq(b)  # Best fit solution
```

### Eigenanalysis

```python
# Compute eigenvalues and eigenvectors
A = Matrix([[1, 2], [2, 1]])
eigenvalues = A.eigenvalues()
eval_vec_pairs = A.eigenvectors()

# Check: Av = λv for each eigenpair
for eval, evec in zip(*eval_vec_pairs):
    Av = A.multiply(evec)
    lambda_v = evec.scale(eval)
    # Av ≈ λv
```

### Matrix Decompositions

```python
# SVD: A = UΣV†
U, sigma, Vh = A.svd()

# QR: A = QR (Q orthogonal, R upper triangular)
Q, R = A.qr()

# LU: A = LU (L lower, U upper triangular)
L, U = A.lu()

# Cholesky: A = LL† (for positive definite A)
L = A.cholesky()
```

### Gram-Schmidt Orthogonalization

```python
# Create orthonormal basis
v1 = Vector([1, 1, 0])
v2 = Vector([1, 0, 1])
v3 = Vector([0, 1, 1])

orthonormal = gram_schmidt([v1, v2, v3])

# Verify orthonormality
for i, vi in enumerate(orthonormal):
    print(f"||v{i}|| = {vi.norm()}")  # Should be 1
    for j, vj in enumerate(orthonormal):
        if i != j:
            print(f"v{i}·v{j} = {vi.dot(vj)}")  # Should be 0
```

### Complex Matrices

```python
# Hermitian matrix
H = Matrix([[1, 1+1j], [1-1j, 2]])
print(H.is_hermitian())  # True

# Unitary matrix (complex orthogonal)
theta = np.pi/4
U = Matrix([[np.cos(theta), -np.sin(theta)*1j],
           [np.sin(theta)*1j, np.cos(theta)]])
print(U.is_orthogonal())  # True (within tolerance)
```

## 🎨 CLI Output Formats

```bash
# Pretty print (default)
vecmatrix matrix inverse '[[1,2],[3,4]]'

# JSON output (for scripting)
vecmatrix matrix eigenvalues '[[1,2],[2,1]]' --format json

# LaTeX output (for papers/presentations)
vecmatrix matrix inverse '[[a,b],[c,d]]' --symbolic --format latex

# Plain text
vecmatrix vector dot '[1,2,3]' '[4,5,6]' --format plain
```

## 🌐 API Server Examples

### Python Client

```python
import requests
import json

url = "http://localhost:8000"

# Vector dot product
response = requests.post(f"{url}/vector/dot", json={
    "args": [[1, 2, 3], [4, 5, 6]]
})
result = response.json()
print(result["result"])  # 32.0

# Matrix inverse
response = requests.post(f"{url}/matrix/inverse", json={
    "args": [[[1, 2], [3, 4]]]
})
result = response.json()
print(result["result"]["data"])  # [[-2, 1], [1.5, -0.5]]

# Symbolic mode
response = requests.post(f"{url}/matrix/det", json={
    "args": [[['a', 'b'], ['c', 'd']]],
    "symbolic": true
})
result = response.json()
print(result["result"])  # "a*d - b*c"
```

### JavaScript Client

```javascript
// Vector operations
const dotProduct = await fetch('http://localhost:8000/vector/dot', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    args: [[1, 2, 3], [4, 5, 6]]
  })
}).then(r => r.json());

console.log(dotProduct.result);  // 32.0

// Matrix operations
const inverse = await fetch('http://localhost:8000/matrix/inverse', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    args: [[[1, 2], [3, 4]]]
  })
}).then(r => r.json());

console.log(inverse.result.data);  // [[-2, 1], [1.5, -0.5]]
```

## 🧪 Testing

```bash
# Run all tests
python test_vecmatrix.py

# Run with pytest (more detailed output)
pytest test_vecmatrix.py -v

# Run with coverage
pytest test_vecmatrix.py --cov=core --cov-report=html
```

## 🚀 Performance Tips

1. **Use Numerical Mode by Default**: Symbolic computation is powerful but slower
2. **Pre-compute Decompositions**: Cache LU, QR, etc. for repeated solves
3. **Batch Operations**: Use matrix-matrix over repeated matrix-vector
4. **Choose Right Solver**: Use specialized solvers (Cholesky for PD matrices)
5. **Avoid Unnecessary Conversions**: Stay in one mode when possible

## 🔒 Thread Safety

All operations are thread-safe. You can safely use VecMatrix in concurrent applications:

```python
from concurrent.futures import ThreadPoolExecutor

def compute_inverse(matrix_data):
    m = Matrix(matrix_data)
    return m.inverse()

with ThreadPoolExecutor(max_workers=4) as executor:
    matrices = [[[1,2],[3,4]], [[5,6],[7,8]], ...]
    results = executor.map(compute_inverse, matrices)
```

## 📝 Error Handling

VecMatrix provides specific exceptions for different error cases:

```python
from core import ModeError, DimensionError, SingularMatrixError

try:
    v1 = Vector([1, 2, 3])
    v2 = Vector(['x', 'y', 'z'], symbolic=True)
    v1.add(v2)  # ModeError
except ModeError as e:
    print(f"Mode mismatch: {e}")

try:
    m = Matrix([[1, 2], [2, 4]])  # Singular
    m.inverse()  # SingularMatrixError
except SingularMatrixError as e:
    print(f"Cannot invert: {e}")
```

## 🤝 Contributing

Contributions welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`python test_vecmatrix.py`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Built on NumPy and SciPy for numerical operations
- Uses SymPy for symbolic mathematics
- Inspired by MATLAB, Mathematica, and modern linear algebra libraries

## 📞 Support

- Email: maesonsfarms@gmail.com (be nice or ignored)

---

**VecMatrix** - The ultimate vector and matrix manipulation library for Python.
Built for researchers, engineers, and developers who demand excellence.
