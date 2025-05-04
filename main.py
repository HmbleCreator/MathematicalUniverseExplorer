import sys
import numpy as np
from equation_parser import EquationParser
from visualization import Visualizer
from ui_handler import UIHandler

class MathematicalUniverseExplorer:
    def __init__(self):
        self.app = UIHandler()
        self.parser = EquationParser()
        self.visualizer = Visualizer()
        self.setup_ui()
        
    def setup_ui(self):
        # Initialize UI components
        self.app.setup_ui("Advanced Mathematical Universe Explorer", (350, 400, 2300, 1200))
        
        # Add equation selection combo box
        self.equation_combo = self.app.add_combo_box(
            "Equation:", 
            ["Sine Wave", "Gaussian", "Hyperbolic Tangent", "Ripple", 
             "Schrödinger Wave Function", "Hydrogen Orbital", 
             "Schwarzschild Metric", "Minkowski Space", "Custom Equation"]
        )
        self.equation_combo.currentIndexChanged.connect(self.update_equation_display)
        
        # Add custom equation input
        self.custom_eq_input = self.app.add_text_input(
            "Custom:", 
            "Enter equation using natural math notation: sin(x), e^x, x^2, ħ, α, σ, ∇², etc.",
            height=100
        )
        
        # Add buttons
        self.help_button = self.app.add_button("Equation Help", self.show_equation_help)
        self.plot_button = self.app.add_button("Plot", self.plot_model)
        self.upload_button = self.app.add_button("Upload Equation", self.upload_equation)
        
        self.app.add_button_row([self.help_button, self.plot_button, self.upload_button])
        
        # Add example equations
        examples = [
            "sin(x) * cos(y)",
            "e^(-x^2 - y^2)",
            "x^2 - y^2",
            "e^(-r^2/2) * cos(5*θ)^2",
            "ħ^2/(2*m) * ∇^2ψ + V(r) * ψ",
            "c^2*dt^2 - dx^2 - dy^2 - dz^2",
            "1 - 2*G*M/(r*c^2)",
            "∫_0^π sin(α*x)*cos(β*y) dα",
            "∑_{i=1}^n x_i^2 + y_i^2"
        ]
        self.app.add_example_group("Example Equations", examples, self.use_example)
        
        # Initialize visualization
        self.canvas = self.visualizer.figure.canvas
        self.app.splitter.addWidget(self.canvas)
        self.app.splitter.setSizes([600, 1700])
        
        # Initialize equation display
        self.update_equation_display()
        
        self.app.show()

    def update_equation_display(self):
        """Update the equation input based on the selected type"""
        selected_eq = self.equation_combo.currentText()
        
        if selected_eq == "Schrödinger Wave Function":
            self.custom_eq_input.setText("-(ħ^2/(2*m)) * ∇²ψ + V * ψ = E * ψ")
        elif selected_eq == "Hydrogen Orbital":
            self.custom_eq_input.setText("exp(-r/(2*a₀)) * (r/a₀)^1 * cos(θ)^2")
        elif selected_eq == "Schwarzschild Metric":
            self.custom_eq_input.setText("-(1 - 2*G*M/(r*c^2)) * c^2*dt^2 + (1 - 2*G*M/(r*c^2))^(-1) * dr^2")
        elif selected_eq == "Minkowski Space":
            self.custom_eq_input.setText("c^2*dt^2 - dx^2 - dy^2 - dz^2")

    def use_example(self, equation):
        """Set the custom equation field to the selected example"""
        self.equation_combo.setCurrentText("Custom Equation")
        self.custom_eq_input.setText(equation)

    def upload_equation(self):
        """Handle equation file upload"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window,
            "Open Equation File",
            "",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    equation = f.read().strip()
                    self.equation_combo.setCurrentText("Custom Equation")
                    self.custom_eq_input.setText(equation)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self.window,
                    "Error",
                    f"Failed to load equation file: {str(e)}"
                )

    def show_equation_help(self):
        """Show help dialog with equation syntax examples"""
        help_text = """
        <h2>Advanced Mathematical Notation Help</h2>
        <p>Enter mathematical expressions using natural notation:</p>

        <h3>Basic Operations:</h3>
        <ul>
            <li>Basic: + - * / ^</li>
            <li>Trig: sin(x), cos(x), tan(x), arcsin(x), arccos(x), arctan(x)</li>
            <li>Hyperbolic: sinh(x), cosh(x), tanh(x)</li>
            <li>Other: exp(x), e^x, log(x), sqrt(x), abs(x)</li>
        </ul>

        <h3>Constants:</h3>
        <ul>
            <li>π, pi - 3.14159...</li>
            <li>e - 2.71828...</li>
            <li>ħ - Reduced Planck constant</li>
            <li>c - Speed of light</li>
            <li>G - Gravitational constant</li>
            <li>α - Fine structure constant</li>
            <li>a₀ - Bohr radius</li>
        </ul>

        <h3>Quantum Mechanics Notation:</h3>
        <ul>
            <li>ψ - Wave function</li>
            <li>|ψ⟩ - Ket vector</li>
            <li>⟨ψ| - Bra vector</li>
            <li>⟨ψ|φ⟩ - Inner product</li>
            <li>Ĥ - Hamiltonian operator</li>
            <li>∇² - Laplacian operator</li>
            <li>σₓ, σᵧ, σᵦ - Pauli matrices</li>
        </ul>

        <h3>Relativity Notation:</h3>
        <ul>
            <li>ημν - Minkowski metric</li>
            <li>gμν - General metric tensor</li>
            <li>∂μ - Partial derivative</li>
            <li>∇μ - Covariant derivative</li>
            <li>Rμν - Ricci tensor</li>
        </ul>

        <h3>Special Characters:</h3>
        <ul>
            <li>Greek letters: α, β, γ, Γ, δ, Δ, ε, θ, λ, μ, ν, π, σ, τ, φ, ψ, ω, Ω</li>
            <li>Subscripts: x₁, x₂, x₃</li>
            <li>Superscripts: x¹, x², x³</li>
            <li>Operators: ∫, ∑, ∏, ∇, √</li>
        </ul>

        <h3>Complex Examples:</h3>
        <ul>
            <li>Schrödinger equation: -(ħ^2/(2*m)) * ∇²ψ + V * ψ = E * ψ</li>
            <li>Schwarzschild metric: -(1 - 2*G*M/(r*c^2)) * c^2*dt^2 + dr^2/(1 - 2*G*M/(r*c^2))</li>
            <li>Heisenberg uncertainty: Δx * Δp ≥ ħ/2</li>
            <li>Wave function: ψ(x) = A * sin(k*x - ω*t)</li>
        </ul>
        """

        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle("Advanced Equation Help")
        msg_box.setTextFormat(QtCore.Qt.RichText)
        msg_box.setText(help_text)
        msg_box.exec_()

    def parse_equation(self, equation_str):
        """
        Parse a human-readable math equation into a numpy-compatible expression
        with advanced mathematical notation support
        """
        # Save original for error messages
        original_eq = equation_str
        
        # First handle special symbols and replace with their Python equivalents
        greek_symbols = {
            # Greek letters
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'Γ': 'Gamma',
            'δ': 'delta', 'Δ': 'Delta', 'ε': 'epsilon', 'ζ': 'zeta',
            'η': 'eta', 'θ': 'theta', 'Θ': 'Theta', 'ι': 'iota',
            'κ': 'kappa', 'λ': 'lambda', 'Λ': 'Lambda', 'μ': 'mu',
            'ν': 'nu', 'ξ': 'xi', 'Ξ': 'Xi', 'π': 'np.pi',
            'ρ': 'rho', 'σ': 'sigma', 'Σ': 'Sigma', 'τ': 'tau',
            'υ': 'upsilon', 'φ': 'phi', 'Φ': 'Phi', 'χ': 'chi',
            'ψ': 'psi', 'Ψ': 'Psi', 'ω': 'omega', 'Ω': 'Omega',
            
            # Special constants
            'ħ': '1.0545718e-34',  # Reduced Planck constant
            'c': '299792458',      # Speed of light
            'G': '6.67430e-11',    # Gravitational constant
            'a₀': '5.29177e-11',   # Bohr radius
            'e': 'np.e',           # Euler's number
            
            # Special operators
            '∇²': 'laplacian',      # Laplacian operator
            '∇': 'grad',            # Gradient operator
            '∫': 'integrate',       # Integral
            '∑': 'sum',             # Summation
            '∏': 'product',         # Product
            
            # Quantum operators
            'σₓ': 'np.array([[0, 1], [1, 0]])',  # Pauli X
            'σᵧ': 'np.array([[0, -1j], [1j, 0]])',  # Pauli Y
            'σᵦ': 'np.array([[1, 0], [0, -1]])',  # Pauli Z
            'Ĥ': 'H',               # Hamiltonian
            
            # Various subscripts
            '₀': '_0', '₁': '_1', '₂': '_2', '₃': '_3', '₄': '_4',
            '₅': '_5', '₆': '_6', '₇': '_7', '₈': '_8', '₉': '_9',
            
            # Various superscripts
            '⁰': '^0', '¹': '^1', '²': '^2', '³': '^3', '⁴': '^4',
            '⁵': '^5', '⁶': '^6', '⁷': '^7', '⁸': '^8', '⁹': '^9',
        }

        # Replace all Greek symbols and special characters
        for symbol, replacement in greek_symbols.items():
            equation_str = equation_str.replace(symbol, replacement)

        # Handle common physics/math constants
        constants = {
            r'\bm\b': 'mass',          # Mass
            r'\br\b': 'np.sqrt(x**2 + y**2)',  # Radial distance
            r'\bθ\b': 'np.arctan2(y, x)',      # Angle theta
            r'\bE\b': 'energy',        # Energy
            r'\bV\b': 'potential',     # Potential
            r'\bt\b': 'time',          # Time
            r'\bR\b': 'radius',        # Radius
            r'\bM\b': 'mass',          # Mass (in GR context)
        }
        
        for pattern, replacement in constants.items():
            equation_str = re.sub(pattern, replacement, equation_str)

        # Replace quantum mechanics specific notations
        quantum_patterns = [
            # Bra-ket notation
            (r'\|([^>]+)\\rangle', r'ket(\1)'),      # |ψ⟩
            (r'\\langle([^|]+)\|', r'bra(\1)'),      # ⟨ψ|
            (r'\\langle([^|]+)\|([^>]+)\\rangle', r'inner_product(\1, \2)'),  # ⟨ψ|φ⟩
            
            # Operators
            (r'\blaplacian\(([^)]+)\)', r'(np.gradient(np.gradient(\1, x), x) + np.gradient(np.gradient(\1, y), y))'),
            (r'\bgrad\(([^)]+)\)', r'np.array([np.gradient(\1, x), np.gradient(\1, y)])'),
        ]
        
        for pattern, replacement in quantum_patterns:
            equation_str = re.sub(pattern, replacement, equation_str)
        
        # General mathematical function replacements
        replacements = [
            # Constants
            (r'\bpi\b', 'np.pi'),
            
            # Trig functions
            (r'\bsin\(', 'np.sin('),
            (r'\bcos\(', 'np.cos('),
            (r'\btan\(', 'np.tan('),
            (r'\barcsin\(', 'np.arcsin('),
            (r'\barccos\(', 'np.arccos('),
            (r'\barctan\(', 'np.arctan('),
            (r'\bsinh\(', 'np.sinh('),
            (r'\bcosh\(', 'np.cosh('),
            (r'\btanh\(', 'np.tanh('),
            
            # Other functions
            (r'\blog\(', 'np.log('),
            (r'\bln\(', 'np.log('),
            (r'\bexp\(', 'np.exp('),
            (r'\bsqrt\(', 'np.sqrt('),
            (r'\babs\(', 'np.abs('),
            (r'\be\^', 'np.exp('),
            
            # Special patterns
            (r'([a-zA-Z0-9_]+)\^([a-zA-Z0-9_]+)', r'\1**\2'),  # x^y -> x**y
            (r'([a-zA-Z0-9_]+)\^([0-9.-]+)', r'\1**\2'),       # x^2 -> x**2
        ]
        
        for pattern, replacement in replacements:
            equation_str = re.sub(pattern, replacement, equation_str)
        
        # Handle division safety
        if '/' in equation_str:
            equation_str = f"np.divide({equation_str}, 1.0, where=np.abs(1.0)>1e-10, out=np.zeros_like({equation_str}))"
        
        # Return the parsed equation
        return equation_str

    def create_visualization(self, x, y, z, viz_type, colormap):
        """Create different types of visualizations"""
        self.figure.clear()
        
        if viz_type == "Surface":
            ax = self.figure.add_subplot(111, projection='3d')
            surf = ax.plot_surface(x, y, z, cmap=colormap, linewidth=0, antialiased=True)
            self.figure.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
        elif viz_type == "Contour":
            ax = self.figure.add_subplot(111)
            contour = ax.contourf(x, y, z, 20, cmap=colormap)
            self.figure.colorbar(contour, ax=ax)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
        elif viz_type == "Wireframe":
            ax = self.figure.add_subplot(111, projection='3d')
            ax.plot_wireframe(x, y, z, rstride=5, cstride=5, color='blue')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
        elif viz_type == "Heat Map":
            ax = self.figure.add_subplot(111)
            heatmap = ax.pcolormesh(x, y, z, cmap=colormap, shading='auto')
            self.figure.colorbar(heatmap, ax=ax)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
        # Update the canvas
        self.canvas.draw()

    def plot_model(self):
        """Plot either built-in or custom equation with advanced visualization options"""
        try:
            # Get domain values
            x_min = float(self.x_min.text())
            x_max = float(self.x_max.text())
            y_min = float(self.y_min.text())
            y_max = float(self.y_max.text())
            resolution = int(self.resolution.text())
            
            x = np.linspace(x_min, x_max, resolution)
            y = np.linspace(y_min, y_max, resolution)
            x, y = np.meshgrid(x, y)
            
            # Get visualization preferences
            viz_type = self.visualization_type.currentText()
            colormap = self.colormap.currentText()
            
            selected_eq = self.equation_combo.currentText()

            if selected_eq == "Sine Wave":
                z = np.sin(np.sqrt(x**2 + y**2))
                title = 'Sine Wave Surface'
            elif selected_eq == "Gaussian":
                z = np.exp(-(x**2 + y**2)/2)
                title = 'Gaussian Surface'
            elif selected_eq == "Hyperbolic Tangent":
                z = np.tanh(np.sqrt(x**2 + y**2))
                title = 'Hyperbolic Tangent Surface'
            elif selected_eq == "Ripple":
                z = np.sin(x**2 + y**2)
                title = 'Ripple Surface'
            elif selected_eq == "Schrödinger Wave Function":
                # Simplified 2D time-independent Schrödinger equation visualization
                # This represents the probability density of a 2D particle in a box
                n_x, n_y = 3, 2  # Quantum numbers
                L_x, L_y = 10, 10  # Box dimensions
                z = np.sin(n_x * np.pi * (x + 5) / L_x) * np.sin(n_y * np.pi * (y + 5) / L_y)
                z = z**2  # Probability density
                title = 'Quantum Particle in a Box (Probability Density)'
            elif selected_eq == "Hydrogen Orbital":
                # Simplified 2D hydrogen orbital visualization
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                # 2p orbital (approximation in 2D)
                z = np.exp(-r/2) * r * np.cos(theta)**2
                title = 'Hydrogen 2p Orbital (cross-section)'
            elif selected_eq == "Schwarzschild Metric":
                # Visualization of gravitational potential in Schwarzschild metric
                G = 6.67430e-11  # Gravitational constant
                M = 1.0          # Mass (in arbitrary units)
                c = 3.0e8        # Speed of light
                r = np.sqrt(x**2 + y**2)
                # Avoid division by zero
                r = np.maximum(r, 0.1)  
                # Gravitational potential term from Schwarzschild metric
                z = 1 - 2*G*M/(r*c**2)
                title = 'Schwarzschild Metric Component (1 - 2GM/rc²)'
            elif selected_eq == "Minkowski Space":
                # Visualization of spacetime interval in 2D Minkowski space
                c = 3.0e8  # Speed of light
                # Creating a visualization where z represents the spacetime interval
                # for fixed dt = 1 (arbitrary units)
                dt = 1.0
                z = c**2 * dt**2 - x**2 - y**2
                title = 'Minkowski Space (Spacetime Interval)'
            else:  # Custom Equation
                try:
                    # Get custom equation
                    custom_eq = self.custom_eq_input.toPlainText()
                    if not custom_eq:
                        raise ValueError("Please enter a custom equation")

                    # Parse the human-friendly equation to numpy format
                    parsed_eq = self.parse_equation(custom_eq)
                    
                    # Create local namespace with safe numpy functions and variables
                    safe_dict = {
                        'np': np,
                        'x': x,
                        'y': y,
                        'mass': 9.1093837e-31,  # electron mass
                        'energy': 1.0,
                        'potential': 0.0,
                        'time': 0.0,
                        'radius': 1.0,
                    }

                    # Evaluate the equation
                    z = eval(parsed_eq, {"__builtins__": {}}, safe_dict)

                    # Handle potential issues with the result
                    if np.any(np.isinf(z)) or np.any(np.isnan(z)):
                        # Replace inf/nan values with finite values for visualization
                        z = np.nan_to_num(z, nan=0.0, posinf=100, neginf=-100)
                        QtWidgets.QMessageBox.warning(self.window, "Warning", 
                            f"The equation produced some infinite or undefined values.\n"
                            f"These have been replaced with finite values for visualization.")

                    title = f'Custom Equation: {custom_eq}'

                except Exception as e:
                    QtWidgets.QMessageBox.critical(self.window, "Error", 
                        f"Error evaluating equation: {str(e)}\n\n"
                        f"Parsed equation: {parsed_eq}\n\n"
                        f"Try using the 'Equation Help' button for syntax examples.")
                    return

            # Create the visualization
            self.create_visualization(x, y, z, viz_type, colormap)
            
            # Set the window title to include the equation
            self.window.setWindowTitle(f"Advanced Mathematical Universe Explorer - {title}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self.window, "Error", 
                f"An error occurred during plotting: {str(e)}\n"
                f"Please check your inputs and try again.")
            return

    def show(self):
        self.window.show()
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    explorer = MathematicalUniverseExplorer()