# Building Executable from ML Text Assistant

This guide explains how to convert the Python application to standalone executable files for Windows and Linux.

## Prerequisites

### For Windows:
1. Python 3.7+ installed
2. Windows 10/11
3. All dependencies installed: `pip install -r requirements.txt`

### For Linux:
1. Python 3.7+ installed
2. Linux distribution (Ubuntu, Debian, Fedora, etc.)
3. All dependencies installed: `pip3 install -r requirements.txt`
4. Required system packages (may vary by distribution):
   - `sudo apt-get install python3-dev` (Ubuntu/Debian)
   - `sudo yum install python3-devel` (Fedora/RHEL)

## Building for Windows

### Method 1: Using Batch Script (Easiest)
1. Double-click `build_exe.bat`
2. Wait for the build to complete (5-10 minutes)
3. The executable will be in: `dist\ML Text Assistant.exe`

### Method 2: Using Python Script
```bash
python build_exe.py
```

### Method 3: Using PowerShell
```powershell
powershell -ExecutionPolicy Bypass -File build_exe.ps1
```

## Building for Linux

### Method 1: Using Bash Script (Easiest)
1. Make the script executable:
   ```bash
   chmod +x build_exe_linux.sh
   ```
2. Run the script:
   ```bash
   ./build_exe_linux.sh
   ```
3. The executable will be in: `dist/ML Text Assistant`
4. Make it executable:
   ```bash
   chmod +x "dist/ML Text Assistant"
   ```

### Method 2: Using Universal Python Script
```bash
python3 build_exe_universal.py
```
This script automatically detects your platform and builds accordingly.

### Method 3: Manual PyInstaller Command
```bash
pip3 install pyinstaller
pyinstaller --name="ML Text Assistant" --onefile --windowed --add-data="rf_task_classifier.pkl:." --hidden-import=webview --hidden-import=sklearn --collect-all=webview --collect-all=sklearn --noconfirm --clean ml-service.py
```

## Important Notes

1. **File Size**: The executable will be large (100-200MB) as it includes Python and all dependencies
2. **Model File**: The model file (`rf_task_classifier.pkl`) is bundled with the executable
3. **First Run**: If the model file doesn't exist, it will be trained on first run (takes a few minutes)
4. **Cross-Platform**: 
   - Windows executables must be built on Windows
   - Linux executables must be built on Linux
   - You cannot build Linux executables on Windows (and vice versa)
5. **Distribution**: 
   - Windows: Distribute the `.exe` file
   - Linux: Distribute the executable file (no extension)

## Linux-Specific Notes

1. **System Dependencies**: Linux requires GUI libraries for the embedded browser window:
   
   **For GTK (Recommended):**
   ```bash
   # Ubuntu/Debian - First update package lists
   sudo apt-get update
   
   # Then try one of these (package names vary by distribution):
   sudo apt-get install python3-gi python3-gi-cairo gir1.2-webkit2-4.1
   # OR
   sudo apt-get install python3-gi python3-gi-cairo gir1.2-webkit2*
   # OR
   sudo apt-get install python3-gi python3-gi-cairo libwebkit2gtk-4.0-dev
   
   # If none work, find available packages:
   apt-cache search gir1.2-webkit
   
   # Fedora/RHEL
   sudo yum install python3-gobject webkitgtk4
   ```
   
   **For QT (Alternative):**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-pyqt5 python3-pyqt5.qtwebengine
   
   # Fedora/RHEL
   sudo yum install python3-qt5
   ```
   
   **Note**: If GUI libraries are not installed, the application will automatically fall back to opening your system's default web browser. The app will still work perfectly, just without the embedded window.

2. **Permissions**: After building, make sure the executable has execute permissions:
   ```bash
   chmod +x "dist/ML Text Assistant"
   ```

3. **Running**: Run the executable directly:
   ```bash
   ./dist/ML\ Text\ Assistant
   ```

4. **Version Warnings**: If you see scikit-learn version warnings, they are harmless. The model will work correctly. To eliminate warnings, retrain the model with the current scikit-learn version.

## Troubleshooting

### Build Fails
- **Windows**: Make sure all dependencies are installed: `pip install -r requirements.txt`
- **Linux**: Make sure all dependencies are installed: `pip3 install -r requirements.txt`
- Make sure PyInstaller is installed: `pip install pyinstaller` or `pip3 install pyinstaller`
- Try running the build command from the project directory

### Executable Doesn't Run
- **Windows**: Check if Windows Defender or antivirus is blocking it
- **Linux**: Make sure the file has execute permissions: `chmod +x "dist/ML Text Assistant"`
- Try running from terminal to see error messages

### Model Not Found Error
- The model will be trained automatically on first run if not found
- This may take a few minutes

### Linux: WebView Not Working
- **The app will automatically fall back to system browser if GUI libraries are missing.**
- To enable embedded window, install required system packages:
  ```bash
  # Ubuntu/Debian (GTK - Recommended)
  sudo apt-get install python3-gi python3-gi-cairo gir1.2-webkit2-4.0
  
  # Fedora/RHEL (GTK)
  sudo yum install python3-gobject webkitgtk4
  
  # Ubuntu/Debian (QT - Alternative)
  sudo apt-get install python3-pyqt5 python3-pyqt5.qtwebengine
  
  # Fedora/RHEL (QT)
  sudo yum install python3-qt5
  ```
- **Note**: The application works perfectly without these libraries - it just opens in your default browser instead of an embedded window.

## Distribution

### Windows:
1. Build the executable using one of the Windows methods
2. Copy `dist\ML Text Assistant.exe` to the target Windows machine
3. Double-click to run

### Linux:
1. Build the executable using one of the Linux methods
2. Copy `dist/ML Text Assistant` to the target Linux machine
3. Make it executable: `chmod +x "ML Text Assistant"`
4. Run: `./ML\ Text\ Assistant`

## Building for Both Platforms

If you need executables for both Windows and Linux:

1. **On Windows machine**: Run `build_exe.bat` to create Windows executable
2. **On Linux machine**: Run `./build_exe_linux.sh` to create Linux executable
3. Distribute the appropriate executable to each platform

The executables are standalone and include all dependencies - no Python installation needed on target machines!

