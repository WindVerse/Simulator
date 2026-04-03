#!/bin/bash
# Wind Visualization System - Setup Script for Linux/macOS

echo "========================================"
echo "Wind Visualization System Setup"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
python3 -m venv .venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "[2/4] Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

echo "[3/4] Upgrading pip..."
pip install --upgrade pip

echo "[4/4] Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "To run the application:"
echo "  1. Activate the environment: source .venv/bin/activate"
echo "  2. Run: python main.py"
echo
echo "Or simply run: ./run.sh"
echo
