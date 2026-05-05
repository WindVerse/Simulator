@echo off
REM Wind Visualization System - Run Script for Windows

REM Use portable Python in project directory
if exist python\python.exe (
    python\python.exe main.py
) else (
    echo ERROR: Portable Python not found in python\ folder
    echo Please ensure the python folder exists with python.exe
    pause
    exit /b 1
)

pause
