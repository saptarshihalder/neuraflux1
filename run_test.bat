@echo off
echo Running Python test script...
echo.

:: Try running with python command
python server\model\test_python.py

:: If that failed, try with python3
if %ERRORLEVEL% NEQ 0 (
  echo Trying with python3 command...
  python3 server\model\test_python.py
)

:: If still failed, try with full path options
if %ERRORLEVEL% NEQ 0 (
  echo Trying system Python paths...
  
  :: Try common Python installation paths
  if exist "C:\Python311\python.exe" (
    echo Using Python 3.11...
    C:\Python311\python.exe server\model\test_python.py
  ) else if exist "C:\Program Files\Python311\python.exe" (
    echo Using Python from Program Files...
    "C:\Program Files\Python311\python.exe" server\model\test_python.py
  ) else if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe" (
    echo Using Python from AppData...
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe" server\model\test_python.py
  )
)

echo.
echo Script execution completed.
echo Press any key to exit...
pause > nul 