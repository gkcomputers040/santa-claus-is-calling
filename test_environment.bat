@echo off
REM Script to test the environment setup with a fresh virtual environment
REM This script will:
REM 1. Create a new virtual environment
REM 2. Install dependencies from requirements.txt
REM 3. Test imports to verify everything works

echo ========================================
echo Testing Santa Claus AI Environment
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

echo [1/5] Creating test virtual environment...
if exist "test_venv" (
    echo Removing existing test_venv...
    rmdir /s /q test_venv
)
python -m venv test_venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    exit /b 1
)

echo [2/5] Activating virtual environment...
call test_venv\Scripts\activate.bat

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [4/5] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    call deactivate
    exit /b 1
)

echo [5/5] Testing imports...
python test_imports.py
if errorlevel 1 (
    echo ERROR: Import test failed
    call deactivate
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS! All dependencies installed correctly
echo ========================================
echo.
echo To use this environment:
echo   1. Run: test_venv\Scripts\activate.bat
echo   2. Create your .env file based on .env.example
echo   3. Run the application
echo.
echo To delete the test environment:
echo   rmdir /s /q test_venv
echo.

call deactivate
pause
