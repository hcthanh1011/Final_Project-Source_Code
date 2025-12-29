@echo off
REM ============================================
REM Face Recognition System - Windows Installer
REM ============================================

echo ============================================
echo   FACE RECOGNITION SYSTEM INSTALLER
echo   Platform: Windows
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo [INFO] Python found!
python --version

REM Create virtual environment
echo.
echo [INFO] Creating virtual environment...
python -m venv venv

REM Activate venv
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo [INFO] Installing dependencies...
pip install -r requirements.txt

REM Create directories
echo [INFO] Creating project directories...
mkdir dataset 2>nul
mkdir models 2>nul
mkdir logs 2>nul
mkdir utils 2>nul

echo.
echo ============================================
echo   INSTALLATION COMPLETE!
echo ============================================
echo.
echo Next steps:
echo   1. Run: venv\Scripts\activate
echo   2. Run: python main_system.py
echo.
echo ============================================

pause
