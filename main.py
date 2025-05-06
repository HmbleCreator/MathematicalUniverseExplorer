import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtCore
import re

class MathematicalUniverseExplorer:
    def __init__(self):  # Fixed constructor name
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = QtWidgets.QMainWindow()
        self.setup_ui()

    def setup_ui(self):
        self.window.setWindowTitle("Mathematical Universe Explorer")
        self.window.setGeometry(350, 400, 2300, 1200)

        # Create main splitter for side panel and canvas
        self.splitter = QtWidgets.QSplitter()
        self.window.setCentralWidget(self.splitter)

        # Create side panel for controls
        self.side_panel = QtWidgets.QWidget()
        self.side_layout = QtWidgets.QVBoxLayout(self.side_panel)

        # Add controls for equation selection
        self.controls = QtWidgets.QFormLayout()

        # Equation selection combo box
        self.equation_combo = QtWidgets.QComboBox()
        self.equation_combo.setMinimumWidth(100)
        self.equation_combo.setMinimumHeight(30)
        self.equation_combo.addItems(["Sine Wave", "Gaussian", "Hyperbolic Tangent", "Ripple", "Custom Equation"])
        self.controls.addRow("Equation:", self.equation_combo)

        # Custom equation input
        self.custom_eq_input = QtWidgets.QLineEdit()
        self.custom_eq_input.setPlaceholderText("Enter equation using natural math notation: sin(x), e^x, x^2, etc.")
        self.custom_eq_input.setMinimumWidth(400)
        self.custom_eq_input.setMinimumHeight(100)
        self.controls.addRow("Custom:", self.custom_eq_input)

        # Add help button
        self.help_button = QtWidgets.QPushButton("Equation Help")
        self.help_button.clicked.connect(self.show_equation_help)
        self.help_button.setMinimumHeight(60)

        # Plot button
        self.plot_button = QtWidgets.QPushButton("Plot")
        self.plot_button.clicked.connect(self.plot_model)
        self.plot_button.setMinimumHeight(60)

        # Add buttons in a horizontal layout
        button_layout = QtWidgets.QHBoxLayout()
        self.help_button.setMinimumWidth(200)
        self.plot_button.setMinimumWidth(200)

        # Add upload button
        self.upload_button = QtWidgets.QPushButton("Upload Equation")
        self.upload_button.setMinimumWidth(200)
        self.upload_button.setMinimumHeight(60)
        self.upload_button.clicked.connect(self.upload_equation)

        button_layout.addWidget(self.help_button)
        button_layout.addWidget(self.plot_button)
        button_layout.addWidget(self.upload_button)
        self.controls.addRow("", button_layout)

        self.side_layout.addLayout(self.controls)

        # Add equation examples
        self.example_group = QtWidgets.QGroupBox("Example Equations")
        example_layout = QtWidgets.QVBoxLayout()
        examples = [
            "sin(x) * cos(y)",
            "e^(-x^2 - y^2)",
            "x^2 - y^2",
            "sin(x^2 + y^2) / (x^2 + y^2 + 0.1)",
            "log(x^2 + y^2 + 1)"
        ]

        for example in examples:
            example_button = QtWidgets.QPushButton(example)
            example_button.clicked.connect(lambda _, eq=example: self.use_example(eq))  # Fixed typo
            example_layout.addWidget(example_button)

        self.example_group.setLayout(example_layout)
        self.side_layout.addWidget(self.example_group)
        self.side_layout.addStretch()

        # Create 3D visualization area
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # Add widgets to splitter
        self.splitter.addWidget(self.side_panel)
        self.splitter.addWidget(self.canvas)

        # Set initial sizes
        self.splitter.setSizes([300, 2000])

        self.show()

    def use_example(self, equation):
        """Set the custom equation field to the selected example"""
        self.equation_combo.setCurrentText("Custom Equation")
        self.custom_eq_input.setText(equation)

    def upload_equation(self):
        """Handle equation file upload"""
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
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
        <h3>Equation Syntax Help</h3>
        <p>Enter mathematical expressions using natural notation:</p>

        <h4>Supported operations:</h4>
        <ul>
            <li>Basic: + - * / ^</li>
            <li>Trig: sin(x), cos(x), tan(x), arcsin(x), arccos(x), arctan(x)</li>
            <li>Hyperbolic: sinh(x), cosh(x), tanh(x)</li>
            <li>Other: exp(x), e^x, log(x), sqrt(x), abs(x)</li>
            <li>Quantum: |ψ⟩ (ket), ⟨ψ| (bra), H (Hadamard), X/Y/Z (Pauli gates)</li>
            <li>Constants: pi, e</li>
        </ul>

        <h4>Examples:</h4>
        <ul>
            <li>sin(x) + cos(y)</li>
            <li>e^(-x^2 - y^2)</li>
            <li>sqrt(x^2 + y^2)</li>
            <li>log(abs(x) + 1)</li>
            <li>sin(pi * x) * cos(pi * y)</li>
            <li>|0⟩⟨0| + |1⟩⟨1| (Projection operators)</li>
            <li>H|0⟩ (Hadamard gate applied to |0⟩)</li>
            <li>X|1⟩ (Pauli X gate applied to |1⟩)</li>
        </ul>
        """

        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle("Equation Help")
        msg_box.setTextFormat(QtCore.Qt.RichText)
        msg_box.setText(help_text)
        msg_box.exec_()  # Fixed typo

    def show(self):
        self.window.show()
        sys.exit(self.app.exec_())

    def parse_equation(self, equation_str):
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
            (r'\^', '*'),
            (r'(\d+)([a-zA-Z])', r'\1\2'),
            (r'\)\s\(', ')('),
            (r'\)([a-zA-Z0-9])', r')\1'),
            # Greek symbols and mathematical operators
            (r'\\sum', 'np.sum'),
            (r'\\prod', 'np.prod'),
            (r'\\Sigma', 'np.sum'),
            (r'\\Pi', 'np.prod'),
            # The following two lines are removed because they cause invalid group reference errors
            # (r'\\Pi\s\(from\s+([^=]+)=([^\s]+)\s+to\s+([^\s]+)\)\s\[([^\]]+)\]', r'np.prod([\5 for \1 in range(\2, \3+1)]), axis=0'),
            # (r'\\Sigma\s\(from\s+([^=]+)=([^\s]+)\s+to\s+([^\s]+)\)\s\[([^\]]+)\]', r'np.sum([\4 for \1 in range(\2, \3+1)]), axis=0'),
            (r'\\times', ''),
            (r'\\div', '/'),
            (r'\\cdot', ''),
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
            # Quantum mechanics notation
            (r'\|([^>]+)\\rangle\b', r'np.kron(\1)'),
            (r'\b\\langle([^|]+)\|', r'np.conj(\1)'),
            (r'\bH\b', 'np.array([[1,1],[1,-1]])/np.sqrt(2)'),
            (r'\bX\b', 'np.array([[0,1],[1,0]])'),
            (r'\bY\b', 'np.array([[0,-1j],[1j,0]])'),
            (r'\bZ\b', 'np.array([[1,0],[0,-1]])'),
            # Special patterns
            (r'e\s\^\s\(([^)]+)\)', r'np.exp(\1)'),
            (r'e\s\^\s([a-zA-Z0-9_]+)', r'np.exp(\1)'),
            (r'e\s\^\s(-?\d+(\.\d+)?)', r'np.exp(\1)'),
        ]

        # Apply all replacements
        for pattern, replacement in replacements:
            equation_str = re.sub(pattern, replacement, equation_str)

        return equation_str

    def plot_model(self):
        """Plot either built-in or custom equation"""
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')

        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        x, y = np.meshgrid(x, y)

        selected_eq = self.equation_combo.currentText()

        if selected_eq == "Sine Wave":
            z = np.sin(np.sqrt(x**2 + y**2))  # Fixed x2/y2 to x**2/y**2
            ax.set_title('Sine Wave Surface')
        elif selected_eq == "Gaussian":
            z = np.exp(-(x**2 + y**2)/2)  # Fixed x2/y2 to x**2/y**2
            ax.set_title('Gaussian Surface')
        elif selected_eq == "Hyperbolic Tangent":
            z = np.tanh(np.sqrt(x**2 + y**2))  # Fixed x2/y2 to x**2/y**2
            ax.set_title('Hyperbolic Tangent Surface')
        elif selected_eq == "Ripple":
            z = np.sin(x**2 + y**2)  # Fixed x2/y2 to x**2/y**2
            ax.set_title('Ripple Surface')
        else:
            try:
                # Get custom equation
                custom_eq = self.custom_eq_input.text()
                if not custom_eq:
                    raise ValueError("Please enter a custom equation")

                # Parse the human-friendly equation to numpy format
                parsed_eq = self.parse_equation(custom_eq)

                # Create local namespace with safe numpy functions and variables
                safe_dict = {
                    'np': np,
                    'x': x,
                    'y': y,
                }

                try:
                    # Evaluate the equation
                    z = eval(parsed_eq, {"builtins": None}, safe_dict)

                    # Handle potential issues with the result
                    if np.any(np.isinf(z)) or np.any(np.isnan(z)):
                        # Replace inf/nan values with finite values for visualization
                        z = np.nan_to_num(z, nan=0.0, posinf=100, neginf=-100)
                        QtWidgets.QMessageBox.warning(self.window, "Warning", 
                            f"The equation produced some infinite or undefined values.\n"
                            f"These have been replaced with finite values for visualization.")

                except Exception as e:
                    QtWidgets.QMessageBox.critical(self.window, "Error", 
                        f"Error evaluating equation: {str(e)}\n\n"
                        f"Parsed equation: {parsed_eq}\n\n"
                        f"Try using the 'Equation Help' button for syntax examples.")
                    return

                ax.set_title(f'Custom Equation: {custom_eq}')

            except Exception as e:
                QtWidgets.QMessageBox.critical(self.window, "Error", 
                    f"Invalid equation: {str(e)}\n"
                    f"Please check your syntax and try again.")
                return

        # Create the surface plot
        surf = ax.plot_surface(x, y, z, cmap='viridis', linewidth=0, antialiased=True)

        # Add a color bar
        self.figure.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Update the canvas
        self.canvas.draw()

if __name__ == "__main__":  # Fixed main guard
    explorer = MathematicalUniverseExplorer()