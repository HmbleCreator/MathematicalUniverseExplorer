import numpy as np
import re

class EquationParser:
    def parse(self, equation_str):
        """
        Parse a human-readable math equation into a numpy-compatible expression
        """
        # Save original for error messages
        original_eq = equation_str
        
        # Replace common math notations
        replacements = [
            # Constants
            (r'\bpi\b', 'np.pi'),
            (r'\be\b', 'np.e'),
            
            # Basic operations
            (r'\^', '**'),
            (r'(\d+)([a-zA-Z])', r'\1*\2'),  # Implicit multiplication: 2x -> 2*x
            (r'\)\s*\(', ')*('),             # Implicit multiplication: (x+1)(y+2) -> (x+1)*(y+2)
            (r'\)([a-zA-Z0-9])', r')*\1'),   # Implicit multiplication: (x+1)y -> (x+1)*y
            
            # Greek symbols and mathematical operators
            (r'\\sum', 'np.sum'),
            (r'\\prod', 'np.prod'),
            (r'\\Sigma', 'np.sum'),
            (r'\\Pi', 'np.prod'),
            (r'\\times', '*'),
            (r'\\div', '/'),
            (r'\\cdot', '*'),
            
            # Trig functions
            (r'\bsin\(', 'np.sin('),
            (r'\bcos\(', 'np.cos('),
            (r'\btan\(', 'np.tan('),
            (r'\barcsin\(', 'np.arcsin('),
            (r'\barccos\(', 'np.arccos('),
            (r'\barctan\(', 'np.arctan('),
            
            # Hyperbolic functions
            (r'\bsinh\(', 'np.sinh('),
            (r'\bcosh\(', 'np.cosh('),
            (r'\btanh\(', 'np.tanh('),
            
            # Other functions
            (r'\blog\(', 'np.log('),
            (r'\bln\(', 'np.log('),
            (r'\bexp\(', 'np.exp('),
            (r'\bsqrt\(', 'np.sqrt('),
            (r'\babs\(', 'np.abs('),
            
            # Special patterns
            (r'e\s*\^\s*\(([^)]+)\)', r'np.exp(\1)'),  # e^(expression)
            (r'e\s*\^\s*([a-zA-Z0-9_]+)', r'np.exp(\1)'),  # e^x
            (r'e\s*\^\s*(-?\d+(\.\d+)?)', r'np.exp(\1)'),  # e^2 or e^-2
        ]
        
        # Apply all replacements
        for pattern, replacement in replacements:
            equation_str = re.sub(pattern, replacement, equation_str)
            
        # Check for division by zero protection for expressions like 1/x
        if '/' in equation_str:
            equation_str = f"np.divide({equation_str}, 1.0)"
            
        return equation_str