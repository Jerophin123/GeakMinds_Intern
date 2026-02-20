@echo off
echo ========================================
echo Building ML Text Assistant Executable
echo ========================================
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
    echo.
)

echo Building executable...
echo This may take several minutes, please wait...
echo.

REM Build the executable with PyInstaller
REM Note: Using semicolon (;) for Windows path separator in --add-data
REM For Linux, use colon (:) instead
pyinstaller --name="ML Text Assistant" --onefile --windowed --icon=NONE --add-data="rf_task_classifier.pkl;." --hidden-import=webview --hidden-import=sklearn --hidden-import=sklearn.ensemble --hidden-import=sklearn.feature_extraction.text --hidden-import=sklearn.pipeline --hidden-import=pandas --hidden-import=numpy --hidden-import=joblib --hidden-import=scipy --hidden-import=scipy.sparse.csgraph._validation --collect-all=webview --collect-all=sklearn --noconfirm --clean ml-service.py

if errorlevel 1 (
    echo.
    echo Build failed! Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo The executable is located in: dist\ML Text Assistant.exe
echo.
echo You can now distribute this .exe file to machines without Python installed.
echo.
echo Note: The .exe file will be large (~100-200MB) as it includes Python and all dependencies.
echo.
pause

