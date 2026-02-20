# Linux Executable - Quick Fix Guide

## The Issue

When running the Linux executable, you may see:
```
[pywebview] GTK cannot be loaded
ModuleNotFoundError: No module named 'gi'
```

This happens because the Linux executable needs system-level GUI libraries.

## Solution Options

### Option 1: Install GUI Libraries (Recommended for Embedded Window)

Install GTK libraries to enable the embedded browser window:

**Ubuntu/Debian:**

First, update your package lists:
```bash
sudo apt-get update
```

Then try one of these (in order):

**Option A - Try version 4.1:**
```bash
sudo apt-get install python3-gi python3-gi-cairo gir1.2-webkit2-4.1
```

**Option B - Auto-select available version:**
```bash
sudo apt-get install python3-gi python3-gi-cairo gir1.2-webkit2*
```

**Option C - Install library directly:**
```bash
sudo apt-get install python3-gi python3-gi-cairo libwebkit2gtk-4.0-dev
```

**Option D - Find the right package:**
```bash
# Run the helper script
chmod +x find_linux_packages.sh
./find_linux_packages.sh

# Or search manually
apt-cache search gir1.2-webkit
```

**Fedora/RHEL:**
```bash
sudo yum install python3-gobject webkitgtk4
```

**Arch Linux:**
```bash
sudo pacman -S python-gobject webkit2gtk
```

After installing, rebuild the executable:
```bash
./build_exe_linux.sh
```

### Option 2: Use System Browser (No Installation Needed)

**The application now automatically falls back to your system browser!**

If GUI libraries are not installed, the app will:
1. Show a helpful error message
2. Automatically open the application in your default web browser
3. Continue running normally

**You don't need to install anything** - just run the executable and it will work!

### Option 3: Install QT Instead of GTK

If you prefer QT over GTK:

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-pyqt5 python3-pyqt5.qtwebengine
```

**Fedora/RHEL:**
```bash
sudo yum install python3-qt5
```

## Version Warning (Harmless)

You may also see scikit-learn version warnings:
```
InconsistentVersionWarning: Trying to unpickle estimator from version 1.6.1 when using version 1.8.0
```

**These warnings are harmless** - the model will work correctly. To eliminate them:
1. Delete `rf_task_classifier.pkl`
2. Run the app - it will retrain with the current scikit-learn version

## Summary

- **With GUI libraries**: Embedded window (like Windows version)
- **Without GUI libraries**: Opens in system browser (still fully functional)
- **Version warnings**: Harmless, can be ignored

The application works perfectly in both cases!

