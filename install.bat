@echo off
echo Installing dependencies...
echo.
echo Step 1: Installing numpy (with pre-built wheel)...
pip install numpy>=2.0.0
echo.
echo Step 2: Installing remaining dependencies...
pip install -r requirements.txt
echo.
echo Installation complete!
pause
