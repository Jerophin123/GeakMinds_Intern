"""
Build script for creating executable from ml-service.py
Run this script to create a standalone .exe file
"""
import PyInstaller.__main__
import os
import sys

def build_exe():
    """Build the executable using PyInstaller"""
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, 'ml-service.py')
    model_path = os.path.join(current_dir, 'rf_task_classifier.pkl')
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print("Warning: rf_task_classifier.pkl not found. The model will be trained on first run.")
    
    # PyInstaller arguments
    args = [
        script_path,
        '--name=ML Text Assistant',
        '--onefile',  # Create a single executable file
        '--windowed',  # No console window (GUI only)
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
    
    # Add model file if it exists (Windows uses semicolon as separator)
    if os.path.exists(model_path):
        if sys.platform == 'win32':
            args.append('--add-data={};.'.format(model_path))
        else:
            args.append('--add-data={}:.'.format(model_path))
    
    print("=" * 50)
    print("Building ML Text Assistant Executable")
    print("=" * 50)
    print()
    print("This may take several minutes...")
    print()
    
    try:
        PyInstaller.__main__.run(args)
        print()
        print("=" * 50)
        print("Build Complete!")
        print("=" * 50)
        print()
        print("The executable is located in: dist\\ML Text Assistant.exe")
        print()
        print("You can now distribute this .exe file to machines without Python installed.")
        print()
    except Exception as e:
        print(f"Error during build: {e}")
        print()
        print("Make sure PyInstaller is installed: pip install pyinstaller")
        sys.exit(1)

if __name__ == '__main__':
    build_exe()

