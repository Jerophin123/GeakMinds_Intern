"""
Universal build script for creating executables on Windows and Linux
Automatically detects the platform and builds accordingly
"""
import PyInstaller.__main__
import os
import sys
import platform

def build_exe():
    """Build the executable using PyInstaller for the current platform"""
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, 'ml-service.py')
    model_path = os.path.join(current_dir, 'rf_task_classifier.pkl')
    
    # Detect platform
    is_windows = platform.system() == 'Windows'
    is_linux = platform.system() == 'Linux'
    is_mac = platform.system() == 'Darwin'
    
    print("=" * 50)
    print("Building ML Text Assistant Executable")
    print(f"Platform: {platform.system()}")
    print("=" * 50)
    print()
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print("Warning: rf_task_classifier.pkl not found. The model will be trained on first run.")
        print()
    
    # PyInstaller arguments (common for all platforms)
    args = [
        script_path,
        '--name=ML Text Assistant',
        '--onefile',  # Create a single executable file
        '--windowed',  # No console window (GUI only) - works on Linux with pywebview
        '--hidden-import=webview',
        '--hidden-import=sklearn',
        '--hidden-import=sklearn.ensemble',
        '--hidden-import=sklearn.feature_extraction.text',
        '--hidden-import=sklearn.pipeline',
        '--hidden-import=pandas',
        '--hidden-import=numpy',
        '--hidden-import=joblib',
        '--hidden-import=scipy',
        '--hidden-import=scipy.sparse.csgraph._validation',
        '--collect-all=webview',
        '--collect-all=sklearn',
        '--noconfirm',  # Overwrite without asking
        '--clean',  # Clean cache before building
    ]
    
    # Add model file with platform-specific path separator
    if os.path.exists(model_path):
        if is_windows:
            # Windows uses semicolon
            args.append(f'--add-data={model_path};.')
        else:
            # Linux/Mac uses colon
            args.append(f'--add-data={model_path}:.')
    
    # Platform-specific adjustments
    if is_linux:
        # Linux-specific options if needed
        pass
    elif is_mac:
        # Mac-specific options if needed
        pass
    
    print("This may take several minutes...")
    print()
    
    try:
        PyInstaller.__main__.run(args)
        print()
        print("=" * 50)
        print("Build Complete!")
        print("=" * 50)
        print()
        
        if is_windows:
            exe_name = "dist\\ML Text Assistant.exe"
        else:
            exe_name = "dist/ML Text Assistant"
            print("To make it executable, run: chmod +x 'dist/ML Text Assistant'")
            print()
        
        print(f"The executable is located in: {exe_name}")
        print()
        print("You can now distribute this executable to machines without Python installed.")
        print()
    except Exception as e:
        print(f"Error during build: {e}")
        print()
        print("Make sure PyInstaller is installed: pip install pyinstaller")
        sys.exit(1)

if __name__ == '__main__':
    build_exe()

