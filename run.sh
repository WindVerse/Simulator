#!/bin/bash
# Wind Visualization System - Run Script for Linux/macOS

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Running setup first..."
    ./setup.sh
    source .venv/bin/activate
fi

# Run the application
python main.py
