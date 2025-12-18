#!/usr/bin/env python3
"""
VecMatrix API Server
====================

A REST API server for vector and matrix operations.
Provides JSON-based interface for integration with other systems.

Usage:
    python api_server.py [--host HOST] [--port PORT]

Endpoints:
    POST /vector/<operation>  - Vector operations
    POST /matrix/<operation>  - Matrix operations
    POST /utility/<operation> - Utility operations
    GET  /operations          - List all available operations
    GET  /health              - Health check

Author: ML
Version: 2.0.0
"""

import sys
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core import (
    Vector, Matrix, VecMatrixError,
    tensor_product, gram_schmidt, matrix_power, matrix_exp, matrix_log
)


class APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for VecMatrix API"""
    
    # Available operations
    VECTOR_OPS = [
        'add', 'subtract', 'scale', 'dot', 'cross', 'norm', 'normalize',
        'project_onto', 'angle_between', 'distance_to', 'is_orthogonal_to'
    ]
    
    MATRIX_OPS = [
        'add', 'subtract', 'multiply', 'transpose', 'conjugate', 'adjoint',
        'det', 'trace', 'rank', 'inverse', 'pseudoinverse', 'eigenvalues',
        'eigenvectors', 'svd', 'qr', 'lu', 'cholesky', 'solve', 'lstsq',
        'nullspace', 'columnspace', 'rowspace', 'condition_number',
        'is_square', 'is_symmetric', 'is_hermitian', 'is_orthogonal',
        'is_positive_definite', 'frobenius_norm', 'identity', 'zeros',
        'ones', 'random'
    ]
    
    UTILITY_OPS = [
        'tensor_product', 'gram_schmidt', 'matrix_power', 'matrix_exp', 'matrix_log'
    ]
    
    def _set_headers(self, status_code: int = 200, content_type: str = 'application/json'):
        """Set response headers"""
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def _send_json(self, data: Dict, status_code: int = 200):
        """Send JSON response"""
        self._set_headers(status_code)
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def _send_error(self, message: str, status_code: int = 400):
        """Send error response"""
        self._send_json({
            'error': message,
            'status': 'error'
        }, status_code)
    
    def _parse_request_body(self) -> Dict:
        """Parse JSON request body"""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        
        body = self.rfile.read(content_length)
        try:
            return json.loads(body.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    
    def _serialize_result(self, result: Any) -> Any:
        """Serialize result for JSON response"""
        if isinstance(result, (Vector, Matrix)):
            return {
                'type': result.__class__.__name__.lower(),
                'mode': result.mode,
                'shape': list(result.shape) if isinstance(result, Matrix) else [len(result)],
                'data': result.to_list()
            }
        elif isinstance(result, list):
            if all(isinstance(x, (Vector, Matrix)) for x in result):
                return [self._serialize_result(x) for x in result]
            elif all(isinstance(x, (int, float, complex)) for x in result):
                return [float(x.real) if x.imag == 0 else {'real': x.real, 'imag': x.imag} 
                       for x in result]
            return result
        elif isinstance(result, dict):
            return {str(k): str(v) for k, v in result.items()}
        elif isinstance(result, complex):
            if result.imag == 0:
                return result.real
            return {'real': result.real, 'imag': result.imag}
        elif isinstance(result, (int, float)):
            return result
        elif isinstance(result, bool):
            return result
        elif isinstance(result, tuple):
            # Handle tuples (like SVD, eigendecomposition)
            return [self._serialize_result(x) for x in result]
        else:
            try:
                import sympy as sp
                if isinstance(result, (sp.Expr, sp.Basic)):
                    return str(result)
            except:
                pass
            return str(result)
    
    def _execute_vector_op(self, operation: str, data: Dict) -> Any:
        """Execute vector operation"""
        symbolic = data.get('symbolic', False)
        args = data.get('args', [])
        
        if operation in ['norm', 'normalize']:
            if len(args) != 1:
                raise ValueError(f"{operation} requires 1 vector")
            v = Vector(args[0], symbolic=symbolic)
            return getattr(v, operation)()
        
        elif operation in ['add', 'subtract', 'dot', 'cross', 'project_onto',
                          'angle_between', 'distance_to', 'is_orthogonal_to']:
            if len(args) != 2:
                raise ValueError(f"{operation} requires 2 vectors")
            v1 = Vector(args[0], symbolic=symbolic)
            v2 = Vector(args[1], symbolic=symbolic)
            return getattr(v1, operation)(v2)
        
        elif operation == 'scale':
            if len(args) != 2:
                raise ValueError("scale requires 1 vector and 1 scalar")
            v = Vector(args[0], symbolic=symbolic)
            scalar = args[1]
            return v.scale(scalar)
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _execute_matrix_op(self, operation: str, data: Dict) -> Any:
        """Execute matrix operation"""
        symbolic = data.get('symbolic', False)
        args = data.get('args', [])
        
        # Factory methods
        if operation in ['identity', 'zeros', 'ones']:
            if operation == 'identity':
                n = args[0]
                return Matrix.identity(n, symbolic=symbolic)
            else:
                rows, cols = args[0], args[1]
                if operation == 'zeros':
                    return Matrix.zeros(rows, cols, symbolic=symbolic)
                else:
                    return Matrix.ones(rows, cols, symbolic=symbolic)
        elif operation == 'random':
            rows, cols = args[0], args[1]
            return Matrix.random(rows, cols)
        
        # Single matrix operations
        elif operation in ['transpose', 'conjugate', 'adjoint', 'det', 'trace',
                          'rank', 'inverse', 'pseudoinverse', 'eigenvalues',
                          'eigenvectors', 'svd', 'qr', 'lu', 'cholesky',
                          'nullspace', 'columnspace', 'rowspace', 'condition_number',
                          'is_square', 'is_symmetric', 'is_hermitian', 'is_orthogonal',
                          'is_positive_definite', 'frobenius_norm']:
            if len(args) != 1:
                raise ValueError(f"{operation} requires 1 matrix")
            m = Matrix(args[0], symbolic=symbolic)
            return getattr(m, operation)()
        
        # Two argument operations
        elif operation in ['add', 'subtract', 'multiply']:
            if len(args) != 2:
                raise ValueError(f"{operation} requires 2 arguments")
            m1 = Matrix(args[0], symbolic=symbolic)
            
            # Try to parse second argument
            try:
                m2 = Matrix(args[1], symbolic=symbolic)
            except:
                try:
                    m2 = Vector(args[1], symbolic=symbolic)
                except:
                    m2 = args[1]  # scalar
            
            return getattr(m1, operation)(m2)
        
        elif operation in ['solve', 'lstsq']:
            if len(args) != 2:
                raise ValueError(f"{operation} requires 1 matrix and 1 vector")
            m = Matrix(args[0], symbolic=symbolic)
            b = Vector(args[1], symbolic=symbolic)
            return getattr(m, operation)(b)
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _execute_utility_op(self, operation: str, data: Dict) -> Any:
        """Execute utility operation"""
        symbolic = data.get('symbolic', False)
        args = data.get('args', [])
        
        if operation == 'tensor_product':
            if len(args) != 2:
                raise ValueError("tensor_product requires 2 vectors")
            v1 = Vector(args[0], symbolic=symbolic)
            v2 = Vector(args[1], symbolic=symbolic)
            return tensor_product(v1, v2)
        
        elif operation == 'gram_schmidt':
            vectors = [Vector(v, symbolic=symbolic) for v in args]
            return gram_schmidt(vectors)
        
        elif operation == 'matrix_power':
            if len(args) != 2:
                raise ValueError("matrix_power requires 1 matrix and 1 integer")
            m = Matrix(args[0], symbolic=symbolic)
            n = int(args[1])
            return matrix_power(m, n)
        
        elif operation == 'matrix_exp':
            if len(args) != 1:
                raise ValueError("matrix_exp requires 1 matrix")
            m = Matrix(args[0], symbolic=symbolic)
            return matrix_exp(m)
        
        elif operation == 'matrix_log':
            if len(args) != 1:
                raise ValueError("matrix_log requires 1 matrix")
            m = Matrix(args[0], symbolic=symbolic)
            return matrix_log(m)
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def do_GET(self):
        """Handle GET requests"""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/health':
            self._send_json({
                'status': 'healthy',
                'version': '2.0.0'
            })
        
        elif path == '/operations':
            self._send_json({
                'vector_operations': self.VECTOR_OPS,
                'matrix_operations': self.MATRIX_OPS,
                'utility_operations': self.UTILITY_OPS
            })
        
        else:
            self._send_error('Not found', 404)
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            parsed = urlparse(self.path)
            path_parts = [p for p in parsed.path.split('/') if p]
            
            if len(path_parts) != 2:
                self._send_error('Invalid path format. Use: /<type>/<operation>')
                return
            
            obj_type, operation = path_parts
            data = self._parse_request_body()
            
            # Execute operation
            if obj_type == 'vector':
                if operation not in self.VECTOR_OPS:
                    self._send_error(f'Unknown vector operation: {operation}')
                    return
                result = self._execute_vector_op(operation, data)
            
            elif obj_type == 'matrix':
                if operation not in self.MATRIX_OPS:
                    self._send_error(f'Unknown matrix operation: {operation}')
                    return
                result = self._execute_matrix_op(operation, data)
            
            elif obj_type == 'utility':
                if operation not in self.UTILITY_OPS:
                    self._send_error(f'Unknown utility operation: {operation}')
                    return
                result = self._execute_utility_op(operation, data)
            
            else:
                self._send_error(f'Unknown type: {obj_type}')
                return
            
            # Serialize and send result
            serialized = self._serialize_result(result)
            self._send_json({
                'status': 'success',
                'result': serialized
            })
        
        except VecMatrixError as e:
            self._send_error(str(e), 400)
        except ValueError as e:
            self._send_error(str(e), 400)
        except Exception as e:
            self._send_error(f'Internal error: {str(e)}', 500)
            traceback.print_exc()
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests (CORS preflight)"""
        self._set_headers()
    
    def log_message(self, format, *args):
        """Custom log message format"""
        print(f"[{self.log_date_time_string()}] {format % args}")


def run_server(host: str = 'localhost', port: int = 8000):
    """Run the API server"""
    server_address = (host, port)
    httpd = HTTPServer(server_address, APIHandler)
    
    print(f"VecMatrix API Server v2.0.0")
    print(f"Listening on http://{host}:{port}")
    print(f"Endpoints:")
    print(f"  GET  /health          - Health check")
    print(f"  GET  /operations      - List available operations")
    print(f"  POST /vector/<op>     - Vector operations")
    print(f"  POST /matrix/<op>     - Matrix operations")
    print(f"  POST /utility/<op>    - Utility operations")
    print(f"\nPress Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='VecMatrix API Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == '__main__':
    main()
