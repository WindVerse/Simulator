#!/usr/bin/env python
"""
Wind Visualization System
Main entry point for the application.

An interactive 3D wind visualization tool with ML-based mesh deformation.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt


def main():
    """Main entry point."""
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Wind Visualization System")
    app.setOrganizationName("WindViz")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Import and create main window
    # (Import here to allow dependency checking first)
    from ui.main_window import MainWindow
    
    window = MainWindow()
    window.show()
    
    # Run event loop
    sys.exit(app.exec_())


def check_dependencies():
    """Check if all required dependencies are installed."""
    missing = []
    optional_missing = []
    
    try:
        import PyQt5
    except ImportError:
        missing.append("PyQt5")
    
    try:
        import OpenGL
    except ImportError:
        missing.append("PyOpenGL")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    # PyTorch is optional - falls back to physics model
    try:
        import torch
    except (ImportError, OSError):
        optional_missing.append("torch (ML model disabled, using physics fallback)")
    
    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nPlease install dependencies using:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    if optional_missing:
        print("Optional dependencies not available:")
        for dep in optional_missing:
            print(f"  - {dep}")
        print()


if __name__ == "__main__":
    # Check dependencies first
    check_dependencies()
    
    # Run application
    main()
