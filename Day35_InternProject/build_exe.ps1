# PowerShell script to build ML Text Assistant executable
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building ML Text Assistant Executable" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if PyInstaller is installed
try {
    python -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing PyInstaller..." -ForegroundColor Yellow
        pip install pyinstaller
        Write-Host ""
    }
} catch {
    Write-Host "Installing PyInstaller..." -ForegroundColor Yellow
    pip install pyinstaller
    Write-Host ""
}

Write-Host "Building executable..." -ForegroundColor Green
Write-Host "This may take several minutes, please wait..." -ForegroundColor Yellow
Write-Host ""

# Build the executable
pyinstaller --name="ML Text Assistant" --onefile --windowed --icon=NONE --add-data="rf_task_classifier.pkl;." --hidden-import=webview --hidden-import=sklearn --hidden-import=sklearn.ensemble --hidden-import=sklearn.feature_extraction.text --hidden-import=sklearn.pipeline --hidden-import=pandas --hidden-import=numpy --hidden-import=joblib --hidden-import=scipy --hidden-import=scipy.sparse.csgraph._validation --collect-all=webview --collect-all=sklearn --noconfirm --clean ml-service.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Build Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "The executable is located in: dist\ML Text Assistant.exe" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "You can now distribute this .exe file to machines without Python installed." -ForegroundColor Green
    Write-Host ""
    Write-Host "Note: The .exe file will be large (~100-200MB) as it includes Python and all dependencies." -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "Build failed! Please check the error messages above." -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to exit"


