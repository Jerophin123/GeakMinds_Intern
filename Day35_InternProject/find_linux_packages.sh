#!/bin/bash

echo "========================================"
echo "Finding WebKit/GTK Packages for Your System"
echo "========================================"
echo ""

echo "Checking for available WebKit packages..."
echo ""

# Try to find WebKit packages
echo "Searching for WebKit2 packages:"
apt-cache search gir1.2-webkit 2>/dev/null | head -10
echo ""

echo "Searching for webkit2gtk packages:"
apt-cache search webkit2gtk 2>/dev/null | head -10
echo ""

echo "Checking installed GTK packages:"
dpkg -l | grep -i "webkit\|gtk" | head -10
echo ""

echo "========================================"
echo "Recommended Installation Commands"
echo "========================================"
echo ""

echo "Try these commands in order:"
echo ""
echo "1. Update package lists:"
echo "   sudo apt-get update"
echo ""
echo "2. Try WebKit2 version 4.1:"
echo "   sudo apt-get install python3-gi python3-gi-cairo gir1.2-webkit2-4.1"
echo ""
echo "3. Or try without version number (auto-select):"
echo "   sudo apt-get install python3-gi python3-gi-cairo gir1.2-webkit2*"
echo ""
echo "4. Or install the library directly:"
echo "   sudo apt-get install python3-gi python3-gi-cairo libwebkit2gtk-4.0-dev"
echo ""
echo "5. Alternative: Use QT instead:"
echo "   sudo apt-get install python3-pyqt5 python3-pyqt5.qtwebengine"
echo ""
echo "========================================"
echo "Note: The app works without these packages!"
echo "It will automatically use your system browser."
echo "========================================"

