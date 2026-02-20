#!/bin/bash

echo "Installing GTK/WebKit dependencies for ML Text Assistant..."
echo ""

sudo apt-get install python3-gi python3-gi-cairo gir1.2-webkit2-4.1

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Installation successful!"
    echo "========================================"
    echo ""
    echo "You can now rebuild the executable to get the embedded window:"
    echo "  ./build_exe_linux.sh"
    echo ""
    echo "Or just run the existing executable - it will use the embedded window now."
else
    echo ""
    echo "Installation failed. The app will still work using your system browser."
fi

