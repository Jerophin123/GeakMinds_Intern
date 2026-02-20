#!/bin/bash

echo "========================================"
echo "Building ML Text Assistant Executable (Linux)"
echo "========================================"
echo ""

# Check for required system dependencies
echo "Checking for required system dependencies..."
MISSING_DEPS=0

# Check for GTK (preferred for pywebview on Linux)
if ! python3 -c "import gi" 2>/dev/null; then
    echo "WARNING: GTK Python bindings (gi) not found."
    echo "  Try installing with:"
    echo "    sudo apt-get update"
    echo "    sudo apt-get install python3-gi python3-gi-cairo gir1.2-webkit2-4.1"
    echo "    # OR: sudo apt-get install python3-gi python3-gi-cairo gir1.2-webkit2*"
    echo "    # OR: sudo apt-get install python3-gi python3-gi-cairo libwebkit2gtk-4.0-dev"
    echo "  Or (Fedora/RHEL): sudo yum install python3-gobject webkitgtk4"
    echo ""
    echo "  To find available packages, run: ./find_linux_packages.sh"
    MISSING_DEPS=1
fi

# Check for PyInstaller
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip3 install pyinstaller
    echo ""
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    echo "NOTE: Missing GUI dependencies will cause the app to fall back to system browser."
    echo "The app will still work, but won't have an embedded window."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    echo ""
fi

echo "Building executable..."
echo "This may take several minutes, please wait..."
echo ""

# Build the executable with PyInstaller
# Note: Using colon (:) for Linux path separator in --add-data
pyinstaller --name="ML Text Assistant" \
    --onefile \
    --windowed \
    --add-data="rf_task_classifier.pkl:." \
    --hidden-import=webview \
    --hidden-import=sklearn \
    --hidden-import=sklearn.ensemble \
    --hidden-import=sklearn.feature_extraction.text \
    --hidden-import=sklearn.pipeline \
    --hidden-import=pandas \
    --hidden-import=numpy \
    --hidden-import=joblib \
    --hidden-import=scipy \
    --hidden-import=scipy.sparse.csgraph._validation \
    --collect-all=webview \
    --collect-all=sklearn \
    --noconfirm \
    --clean \
    ml-service.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Build Complete!"
    echo "========================================"
    echo ""
    echo "The executable is located in: dist/ML Text Assistant"
    echo ""
    echo "You can now distribute this executable to Linux machines without Python installed."
    echo ""
    echo "Note: The executable will be large (~100-200MB) as it includes Python and all dependencies."
    echo ""
    echo "To make it executable, run: chmod +x 'dist/ML Text Assistant'"
    echo ""
else
    echo ""
    echo "Build failed! Please check the error messages above."
    exit 1
fi

