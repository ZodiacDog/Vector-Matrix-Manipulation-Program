#!/usr/bin/env python3
"""
VecMatrix CLI - Command Line Interface
=======================================

A powerful command-line interface for vector and matrix operations.

Usage:
    vecmatrix <type> <operation> [arguments] [options]

Examples:
    vecmatrix vector dot '[1,2,3]' '[4,5,6]'
    vecmatrix matrix inverse '[[1,2],[3,4]]'
    vecmatrix vector add '[1,2]' '[3,4]' --symbolic
    vecmatrix matrix eigenvalues '[[1,2],[3,4]]' --format json

Author: ML
Version: 2.0.0
"""

import sys
import json
import argparse
from typing import Any, List, Dict
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core import (
    Vector, Matrix, VecMatrixError,
    tensor_product, gram_schmidt, matrix_power, matrix_exp, matrix_log
)


class CLIFormatter:
    """Handles output formatting for CLI"""
    
    @staticmethod
    def format_output(result: Any, format_type: str = 'pretty') -> str:
        """
        Format output based on type.
        
        Args:
            result: The result to format
            format_type: 'pretty', 'json', 'latex', or 'plain'
        """
        if format_type == 'json':
            return CLIFormatter._to_json(result)
        elif format_type == 'latex':
            return CLIFormatter._to_latex(result)
        elif format_type == 'plain':
            return str(result)
        else:  # pretty
            return CLIFormatter._to_pretty(result)
    
    @staticmethod
    def _to_json(result: Any) -> str:
        """Convert result to JSON"""
        if isinstance(result, (Vector, Matrix)):
            data = {
                'type': result.__class__.__name__.lower(),
                'mode': result.mode,
                'shape': result.shape if isinstance(result, Matrix) else (len(result),),
                'data': result.to_list()
            }
            return json.dumps(data, indent=2)
        elif isinstance(result, (list, tuple)):
            if all(isinstance(x, (Vector, Matrix)) for x in result):
                return json.dumps([CLIFormatter._to_json(x) for x in result])
            return json.dumps(result)
        elif isinstance(result, dict):
            return json.dumps({str(k): str(v) for k, v in result.items()}, indent=2)
        else:
            return json.dumps({'result': str(result)})
    
    @staticmethod
    def _to_latex(result: Any) -> str:
        """Convert result to LaTeX"""
        if isinstance(result, Vector):
            if result.mode == 'symbolic':
                import sympy as sp
                return sp.latex(result.data)
            else:
                # Format as column vector
                entries = ' \\\\ '.join(str(x) for x in result.data)
                return f"\\begin{{pmatrix}} {entries} \\end{{pmatrix}}"
        elif isinstance(result, Matrix):
            if result.mode == 'symbolic':
                import sympy as sp
                return sp.latex(result.data)
            else:
                rows = []
                for i in range(result.shape[0]):
                    row = ' & '.join(str(result.data[i, j]) for j in range(result.shape[1]))
                    rows.append(row)
                return f"\\begin{{pmatrix}} {' \\\\ '.join(rows)} \\end{{pmatrix}}"
        else:
            return str(result)
    
    @staticmethod
    def _to_pretty(result: Any) -> str:
        """Pretty print result with formatting"""
        if isinstance(result, (Vector, Matrix)):
            header = f"=== {result.__class__.__name__} ({result.mode} mode) ==="
            if isinstance(result, Matrix):
                header += f" {result.shape[0]}x{result.shape[1]}"
            else:
                header += f" length {len(result)}"
            return f"{header}\n{str(result)}"
        elif isinstance(result, list):
            if all(isinstance(x, (Vector, Matrix)) for x in result):
                outputs = [CLIFormatter._to_pretty(x) for x in result]
                return '\n\n'.join(outputs)
            return '\n'.join(str(x) for x in result)
        elif isinstance(result, dict):
            lines = [f"{k}: {v}" for k, v in result.items()]
            return '\n'.join(lines)
        else:
            return f"Result: {result}"


class VecMatrixCLI:
    """Main CLI handler"""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description='VecMatrix - Advanced Vector and Matrix Operations',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Vector operations
  vecmatrix vector dot '[1,2,3]' '[4,5,6]'
  vecmatrix vector cross '[1,0,0]' '[0,1,0]'
  vecmatrix vector norm '[3,4]'
  
  # Matrix operations
  vecmatrix matrix multiply '[[1,2],[3,4]]' '[[5,6],[7,8]]'
  vecmatrix matrix inverse '[[1,2],[3,4]]'
  vecmatrix matrix eigenvalues '[[1,2],[2,1]]'
  
  # Advanced operations
  vecmatrix matrix svd '[[1,2],[3,4],[5,6]]'
  vecmatrix matrix qr '[[1,2],[3,4],[5,6]]'
  
  # Symbolic mode
  vecmatrix vector dot '[x,y,z]' '[a,b,c]' --symbolic
  vecmatrix matrix det '[[a,b],[c,d]]' --symbolic
  
  # Output formats
  vecmatrix vector add '[1,2]' '[3,4]' --format json
  vecmatrix matrix inverse '[[1,2],[3,4]]' --format latex
            """
        )
        
        parser.add_argument('type', choices=['vector', 'matrix', 'utility'],
                          help='Object type')
        parser.add_argument('operation', help='Operation to perform')
        parser.add_argument('args', nargs='*', help='Operation arguments (JSON format)')
        
        parser.add_argument('--symbolic', '-s', action='store_true',
                          help='Use symbolic mode')
        parser.add_argument('--format', '-f', choices=['pretty', 'json', 'latex', 'plain'],
                          default='pretty', help='Output format')
        parser.add_argument('--precision', '-p', type=int, default=6,
                          help='Decimal precision for output')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Verbose output')
        
        return parser
    
    def parse_json_arg(self, arg: str) -> Any:
        """Parse JSON argument, handling symbolic expressions"""
        try:
            data = json.loads(arg)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON argument: {arg}\nError: {e}")
    
    def execute_vector_operation(self, operation: str, args: List[str], 
                                 symbolic: bool) -> Any:
        """Execute vector operation"""
        parsed_args = [self.parse_json_arg(arg) for arg in args]
        
        # Single vector operations
        if operation in ['norm', 'normalize']:
            if len(parsed_args) != 1:
                raise ValueError(f"{operation} requires exactly 1 vector")
            v = Vector(parsed_args[0], symbolic=symbolic)
            return getattr(v, operation)()
        
        # Two vector operations
        elif operation in ['add', 'subtract', 'dot', 'cross', 'project_onto', 
                          'angle_between', 'distance_to', 'is_orthogonal_to']:
            if len(parsed_args) != 2:
                raise ValueError(f"{operation} requires exactly 2 vectors")
            v1 = Vector(parsed_args[0], symbolic=symbolic)
            v2 = Vector(parsed_args[1], symbolic=symbolic)
            return getattr(v1, operation)(v2)
        
        # Scale operation
        elif operation == 'scale':
            if len(parsed_args) != 2:
                raise ValueError("scale requires 1 vector and 1 scalar")
            v = Vector(parsed_args[0], symbolic=symbolic)
            scalar = parsed_args[1]
            return v.scale(scalar)
        
        else:
            raise ValueError(f"Unknown vector operation: {operation}")
    
    def execute_matrix_operation(self, operation: str, args: List[str],
                                symbolic: bool) -> Any:
        """Execute matrix operation"""
        parsed_args = [self.parse_json_arg(arg) for arg in args]
        
        # Factory methods
        if operation == 'identity':
            n = parsed_args[0]
            return Matrix.identity(n, symbolic=symbolic)
        elif operation == 'zeros':
            rows, cols = parsed_args[0], parsed_args[1]
            return Matrix.zeros(rows, cols, symbolic=symbolic)
        elif operation == 'ones':
            rows, cols = parsed_args[0], parsed_args[1]
            return Matrix.ones(rows, cols, symbolic=symbolic)
        elif operation == 'random':
            rows, cols = parsed_args[0], parsed_args[1]
            return Matrix.random(rows, cols)
        
        # Single matrix operations
        elif operation in ['transpose', 'conjugate', 'adjoint', 'det', 'trace', 
                          'rank', 'inverse', 'pseudoinverse', 'eigenvalues', 
                          'eigenvectors', 'svd', 'qr', 'lu', 'cholesky',
                          'nullspace', 'columnspace', 'rowspace', 'condition_number',
                          'is_square', 'is_symmetric', 'is_hermitian', 'is_orthogonal',
                          'is_positive_definite', 'frobenius_norm']:
            if len(parsed_args) != 1:
                raise ValueError(f"{operation} requires exactly 1 matrix")
            m = Matrix(parsed_args[0], symbolic=symbolic)
            return getattr(m, operation)()
        
        # Two matrix operations
        elif operation in ['add', 'subtract', 'multiply']:
            if len(parsed_args) != 2:
                raise ValueError(f"{operation} requires exactly 2 arguments")
            m1 = Matrix(parsed_args[0], symbolic=symbolic)
            
            # Check if second arg is matrix, vector, or scalar
            try:
                m2 = Matrix(parsed_args[1], symbolic=symbolic)
            except:
                try:
                    m2 = Vector(parsed_args[1], symbolic=symbolic)
                except:
                    m2 = parsed_args[1]  # scalar
            
            return getattr(m1, operation)(m2)
        
        # Solve operation
        elif operation == 'solve':
            if len(parsed_args) != 2:
                raise ValueError("solve requires 1 matrix and 1 vector")
            m = Matrix(parsed_args[0], symbolic=symbolic)
            b = Vector(parsed_args[1], symbolic=symbolic)
            return m.solve(b)
        
        # Least squares
        elif operation == 'lstsq':
            if len(parsed_args) != 2:
                raise ValueError("lstsq requires 1 matrix and 1 vector")
            m = Matrix(parsed_args[0], symbolic=symbolic)
            b = Vector(parsed_args[1], symbolic=symbolic)
            return m.lstsq(b)
        
        else:
            raise ValueError(f"Unknown matrix operation: {operation}")
    
    def execute_utility_operation(self, operation: str, args: List[str],
                                  symbolic: bool) -> Any:
        """Execute utility operation"""
        parsed_args = [self.parse_json_arg(arg) for arg in args]
        
        if operation == 'tensor_product':
            if len(parsed_args) != 2:
                raise ValueError("tensor_product requires 2 vectors")
            v1 = Vector(parsed_args[0], symbolic=symbolic)
            v2 = Vector(parsed_args[1], symbolic=symbolic)
            return tensor_product(v1, v2)
        
        elif operation == 'gram_schmidt':
            vectors = [Vector(data, symbolic=symbolic) for data in parsed_args]
            return gram_schmidt(vectors)
        
        elif operation == 'matrix_power':
            if len(parsed_args) != 2:
                raise ValueError("matrix_power requires 1 matrix and 1 integer")
            m = Matrix(parsed_args[0], symbolic=symbolic)
            n = int(parsed_args[1])
            return matrix_power(m, n)
        
        elif operation == 'matrix_exp':
            if len(parsed_args) != 1:
                raise ValueError("matrix_exp requires 1 matrix")
            m = Matrix(parsed_args[0], symbolic=symbolic)
            return matrix_exp(m)
        
        elif operation == 'matrix_log':
            if len(parsed_args) != 1:
                raise ValueError("matrix_log requires 1 matrix")
            m = Matrix(parsed_args[0], symbolic=symbolic)
            return matrix_log(m)
        
        else:
            raise ValueError(f"Unknown utility operation: {operation}")
    
    def run(self, args: List[str] = None):
        """Run the CLI"""
        try:
            parsed = self.parser.parse_args(args)
            
            # Execute operation
            if parsed.type == 'vector':
                result = self.execute_vector_operation(
                    parsed.operation, parsed.args, parsed.symbolic
                )
            elif parsed.type == 'matrix':
                result = self.execute_matrix_operation(
                    parsed.operation, parsed.args, parsed.symbolic
                )
            elif parsed.type == 'utility':
                result = self.execute_utility_operation(
                    parsed.operation, parsed.args, parsed.symbolic
                )
            else:
                raise ValueError(f"Unknown type: {parsed.type}")
            
            # Format and print output
            output = CLIFormatter.format_output(result, parsed.format)
            print(output)
            
            return 0
            
        except VecMatrixError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            if parsed.verbose if 'parsed' in locals() else False:
                import traceback
                traceback.print_exc()
            return 2


def main():
    """Main entry point"""
    cli = VecMatrixCLI()
    sys.exit(cli.run())


if __name__ == '__main__':
    main()
