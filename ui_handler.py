from PyQt5 import QtWidgets, QtCore

class UIHandler:
    def __init__(self):
        self.window = QtWidgets.QMainWindow()
        self.splitter = QtWidgets.QSplitter()
        self.side_panel = QtWidgets.QWidget()
        self.side_layout = QtWidgets.QVBoxLayout(self.side_panel)
        self.controls = QtWidgets.QFormLayout()
        
    def setup_ui(self, window_title, geometry):
        """Initialize the main UI components"""
        self.window.setWindowTitle(window_title)
        self.window.setGeometry(*geometry)
        self.window.setCentralWidget(self.splitter)
        
        # Add widgets to splitter
        self.splitter.addWidget(self.side_panel)
        
        # Add controls to side panel
        self.side_layout.addLayout(self.controls)
        
    def add_combo_box(self, label, items, width=100, height=30):
        """Add a combo box to the UI"""
        combo = QtWidgets.QComboBox()
        combo.setMinimumWidth(width)
        combo.setMinimumHeight(height)
        combo.addItems(items)
        self.controls.addRow(label, combo)
        return combo
        
    def add_text_input(self, label, placeholder="", width=400, height=100):
        """Add a text input field to the UI"""
        text_input = QtWidgets.QLineEdit()
        text_input.setPlaceholderText(placeholder)
        text_input.setMinimumWidth(width)
        text_input.setMinimumHeight(height)
        self.controls.addRow(label, text_input)
        return text_input
        
    def add_button(self, text, callback, width=200, height=60):
        """Add a button to the UI"""
        button = QtWidgets.QPushButton(text)
        button.setMinimumWidth(width)
        button.setMinimumHeight(height)
        button.clicked.connect(callback)
        return button
        
    def add_button_row(self, buttons):
        """Add a row of buttons to the UI"""
        button_layout = QtWidgets.QHBoxLayout()
        for button in buttons:
            button_layout.addWidget(button)
        self.controls.addRow("", button_layout)
        
    def add_example_group(self, title, examples, callback):
        """Add a group of example buttons"""
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout()
        
        for example in examples:
            button = QtWidgets.QPushButton(example)
            button.clicked.connect(lambda _, eq=example: callback(eq))
            layout.addWidget(button)
            
        group.setLayout(layout)
        self.side_layout.addWidget(group)
        
    def show(self):
        """Show the main window"""
        self.window.show()