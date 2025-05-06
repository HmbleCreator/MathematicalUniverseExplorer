import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtCore
import re
from equation_parser import parse_equation
from visualization import plot_3d

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

        # Variable input field
        self.variable_input = QtWidgets.QLineEdit()
        self.variable_input.setPlaceholderText("Variables (comma separated, e.g. x, y, t)")
        self.variable_input.setText("x, y")
        self.controls.addRow("Variables:", self.variable_input)

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
            # Classic surfaces
            "sin(x) * cos(y)",
            "e^(-x^2 - y^2)",
            "x^2 - y^2",
            "sin(x^2 + y^2) / (x^2 + y^2 + 0.1)",
            "log(x^2 + y^2 + 1)",
            # Interesting functions
            "sigmoid(x + y)",
            "floor(sin(x) + cos(y))",
            "abs(x*y) / (1 + x^2 + y^2)",
            # Greek and LaTeX
            r"\sin(\pi x) + \cos(\pi y)",
            r"\frac{1}{1 + e^{-x-y}}",  # LaTeX sigmoid
            # Capital Pi/Sigma
            r"Sum_{k=1}^{5} x^k / k!",  # Notation for sum
            # Quantum/Relativity inspired
            r"sqrt(x**2 + y**2)",
            r"exp(-x**2 - y**2) * cos(2*pi*x) * cos(2*pi*y)",
            r"Heaviside(x-y)",
            r"tanh(x*y)",
            r"x*y/(x**2 + y**2 + 1)",
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

    def plot_model(self):
        """Plot either built-in or custom equation"""
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        # Get variables from UI
        variables = tuple(v.strip() for v in self.variable_input.text().split(',') if v.strip())
        if not variables:
            QtWidgets.QMessageBox.critical(self.window, "Error", "Please specify at least one variable.")
            return
        selected_eq = self.equation_combo.currentText()
        if selected_eq == "Sine Wave":
            func = lambda x, y: np.sin(np.sqrt(x**2 + y**2))
        elif selected_eq == "Gaussian":
            func = lambda x, y: np.exp(-(x**2 + y**2)/2)
        elif selected_eq == "Hyperbolic Tangent":
            func = lambda x, y: np.tanh(np.sqrt(x**2 + y**2))
        elif selected_eq == "Ripple":
            func = lambda x, y: np.sin(x**2 + y**2)
        else:
            custom_eq = self.custom_eq_input.text()
            if not custom_eq:
                QtWidgets.QMessageBox.critical(self.window, "Error", "Please enter a custom equation")
                return
            try:
                func = parse_equation(custom_eq, variables=variables)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self.window, "Error", f"Invalid equation: {str(e)}\nPlease check your syntax and try again.")
                return
        # Prepare meshgrid for up to 2 variables
        if len(variables) == 1:
            x = np.linspace(-5, 5, 100)
            y = np.zeros_like(x)
            Z = func(x)
            ax.plot(x, Z)
        elif len(variables) >= 2:
            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x, y)
            try:
                Z = func(X, Y)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self.window, "Error", f"Error evaluating equation: {str(e)}")
                return
            plot_3d(lambda x, y: func(x, y), ax=ax)
        else:
            QtWidgets.QMessageBox.critical(self.window, "Error", "Only 1D and 2D visualizations are supported.")
            return
        self.canvas.draw()

if __name__ == "__main__":  # Fixed main guard
    explorer = MathematicalUniverseExplorer()