# VecMatrix Quick Start Guide

## Installation

```bash
# 1. Navigate to the vecmatrix directory
cd vecmatrix

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install package (optional - for system-wide use)
pip install -e .
```

## Running Tests

```bash
# Run all tests
python test_vecmatrix.py

# Expected output: "Ran 61 tests in X.XXXs - OK"
```

## Quick Examples

### Using as a Python Library

```python
# Import the library
from core import Vector, Matrix

# Vector operations
v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])
print(f"Dot product: {v1.dot(v2)}")  # 32.0

# Matrix operations
A = Matrix([[1, 2], [3, 4]])
print(f"Determinant: {A.det()}")  # -2.0
print(f"Inverse:\n{A.inverse()}")

# Solve Ax = b
b = Vector([5, 11])
x = A.solve(b)
print(f"Solution: {x}")  # [1, 2]
```

### Using the Command Line

```bash
# Vector dot product
python cli.py vector dot '[1,2,3]' '[4,5,6]'

# Matrix inverse
python cli.py matrix inverse '[[1,2],[3,4]]'

# Symbolic mode
python cli.py matrix det '[[a,b],[c,d]]' --symbolic

# JSON output
python cli.py matrix eigenvalues '[[1,2],[2,1]]' --format json
```

### Starting the API Server

```bash
# Start server on localhost:8000
python api_server.py

# In another terminal, test with curl:
curl -X POST http://localhost:8000/vector/dot \
  -H "Content-Type: application/json" \
  -d '{"args": [[1,2,3], [4,5,6]]}'
```

### Running Examples

```bash
# See comprehensive examples
python examples.py
```

## File Structure

```
vecmatrix/
├── core.py              # Main library with Vector and Matrix classes
├── cli.py               # Command-line interface
├── api_server.py        # REST API server
├── test_vecmatrix.py    # Comprehensive test suite
├── examples.py          # Example usage scripts
├── __init__.py          # Package initialization
├── setup.py             # Installation script
├── requirements.txt     # Dependencies
├── README.md            # Full documentation
├── LICENSE              # MIT License
├── Makefile             # Convenience commands
└── QUICKSTART.md        # This file
```

## Key Features

✓ **60+ Operations**: Comprehensive vector and matrix operations
✓ **Dual Mode**: Numerical (NumPy) and Symbolic (SymPy) computation
✓ **Decompositions**: SVD, QR, LU, Cholesky
✓ **Eigenanalysis**: Eigenvalues, eigenvectors
✓ **System Solving**: Direct solve, least squares
✓ **Multiple Interfaces**: Library, CLI, REST API
✓ **Type Safe**: Full validation and error handling
✓ **Production Ready**: Thread-safe, tested, optimized

## Common Operations

### Vectors
- `add`, `subtract`, `scale`: Basic arithmetic
- `dot`, `cross`: Products
- `norm`, `normalize`: Magnitude operations
- `project_onto`, `angle_between`: Geometric operations

### Matrices
- `multiply`, `transpose`, `inverse`: Basic operations
- `det`, `trace`, `rank`: Properties
- `eigenvalues`, `eigenvectors`: Spectral analysis
- `svd`, `qr`, `lu`, `cholesky`: Decompositions
- `solve`, `lstsq`: System solving
- `nullspace`, `columnspace`: Subspaces

## Need Help?

1. See full documentation: `README.md`
2. Run examples: `python examples.py`
3. Check tests for usage patterns: `test_vecmatrix.py`
4. API reference in docstrings: `help(Vector)`, `help(Matrix)`

## Tips

- Use numerical mode (default) for performance
- Use symbolic mode for algebraic manipulation
- Check matrix properties before operations (is_symmetric, is_positive_definite, etc.)
- Use appropriate solvers (Cholesky for PD, lstsq for overdetermined)
- Set precision in CLI with `--precision N`

Enjoy using VecMatrix! 🚀
