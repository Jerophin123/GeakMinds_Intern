# Quick Build Reference

## Windows Build
```bash
# Option 1: Double-click
build_exe.bat

# Option 2: Command line
python build_exe.py

# Option 3: PowerShell
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```

**Output**: `dist\ML Text Assistant.exe`

## Linux Build
```bash
# Option 1: Bash script
chmod +x build_exe_linux.sh
./build_exe_linux.sh

# Option 2: Universal Python script
python3 build_exe_universal.py

# Option 3: Manual
pyinstaller --name="ML Text Assistant" --onefile --windowed --add-data="rf_task_classifier.pkl:." --hidden-import=webview --hidden-import=sklearn --collect-all=webview --collect-all=sklearn --noconfirm --clean ml-service.py
chmod +x "dist/ML Text Assistant"
```

**Output**: `dist/ML Text Assistant`

**Note**: For embedded window, install GUI libraries:
- Ubuntu/Debian: `sudo apt-get install python3-gi python3-gi-cairo gir1.2-webkit2-4.0`
- Without GUI libs, app automatically opens in system browser (still works!)

## Key Differences

| Platform | Path Separator | Executable Name | Notes |
|----------|----------------|-----------------|-------|
| Windows  | `;` (semicolon) | `ML Text Assistant.exe` | Built on Windows |
| Linux    | `:` (colon)     | `ML Text Assistant` | Built on Linux, needs `chmod +x` |

## Important
- **Windows executables must be built on Windows**
- **Linux executables must be built on Linux**
- Both are standalone - no Python needed on target machines

