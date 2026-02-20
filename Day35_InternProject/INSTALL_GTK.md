# Installing GTK/WebKit for Linux - Step by Step

## The Problem

The package `gir1.2-webkit2-4.0` may not be available on your system. Package names and versions vary by Linux distribution.

## Solution: Try These Commands in Order

### Step 1: Update Package Lists
```bash
sudo apt-get update
```

### Step 2: Try Different Package Names

**Option A - Try version 4.1:**
```bash
sudo apt-get install python3-gi python3-gi-cairo gir1.2-webkit2-4.1
```

**Option B - Auto-select available version (wildcard):**
```bash
sudo apt-get install python3-gi python3-gi-cairo gir1.2-webkit2*
```

**Option C - Install library directly:**
```bash
sudo apt-get install python3-gi python3-gi-cairo libwebkit2gtk-4.0-dev
```

**Option D - Find what's available:**
```bash
# Search for WebKit packages
apt-cache search gir1.2-webkit

# Or use the helper script
chmod +x find_linux_packages.sh
./find_linux_packages.sh
```

### Step 3: Alternative - Use QT Instead

If GTK/WebKit doesn't work, try QT:
```bash
sudo apt-get install python3-pyqt5 python3-pyqt5.qtwebengine
```

## Important Note

**You don't need to install these packages!**

The application will work perfectly without them - it will just open in your system's default web browser instead of an embedded window. The functionality is identical.

## After Installation

If you successfully install the packages, rebuild the executable:
```bash
./build_exe_linux.sh
```

Then run:
```bash
./dist/ML\ Text\ Assistant
```

## Still Having Issues?

1. Check your Linux distribution:
   ```bash
   lsb_release -a
   ```

2. Check available packages:
   ```bash
   apt-cache search webkit | grep -i gir
   ```

3. Use the helper script:
   ```bash
   chmod +x find_linux_packages.sh
   ./find_linux_packages.sh
   ```

4. Remember: The app works without these packages! Just run it and it will use your browser.

