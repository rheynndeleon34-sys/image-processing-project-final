#!/usr/bin/env python3
"""
Find all imports in the project to create complete requirements.txt
"""
import os
import re
import sys
from pathlib import Path

def find_all_imports():
    """Find all import statements in Python files"""
    project_root = Path(".")
    
    # Standard library modules (don't include in requirements.txt)
    stdlib_modules = {
        # Python standard library
        'os', 'sys', 're', 'math', 'json', 'time', 'datetime', 'collections',
        'itertools', 'functools', 'typing', 'pathlib', 'random', 'statistics',
        'string', 'hashlib', 'base64', 'csv', 'html', 'xml', 'uuid',
        'subprocess', 'multiprocessing', 'threading', 'queue', 'socket',
        'ssl', 'email', 'http', 'urllib', 'logging', 'argparse', 'getopt',
        'inspect', 'pdb', 'unittest', 'doctest', 'pickle', 'shelve', 'sqlite3',
        'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile', 'shutil',
        'glob', 'fnmatch', 'linecache', 'codecs', 'unicodedata', 'stringprep',
        'textwrap', 'difflib', 'pprint', 'reprlib', 'enum', 'numbers',
        'decimal', 'fractions', 'array', 'memoryview', 'ctypes', 'mmap',
        'readline', 'rlcompleter', 'atexit', 'signal', 'traceback', 'errno',
        'io', 'termios', 'tty', 'curses', 'platform', 'sysconfig', 'site',
        'code', 'codeop', 'zipimport', 'pkgutil', 'modulefinder', 'runpy',
        'importlib', 'parser', 'ast', 'symtable', 'symbol', 'token', 'keyword',
        'tokenize', 'tabnanny', 'py_compile', 'pyclbr', 'compileall', 'dis',
        'pickletools', 'formatter', 'msilib', 'msvcrt', 'winsound', 'posix',
        'pwd', 'spwd', 'grp', 'crypt', 'termios', 'tty', 'pty', 'fcntl',
        'pipes', 'resource', 'nis', 'syslog', 'optparse', 'imp',
    }
    
    # Map import names to pip package names
    pip_name_map = {
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'skimage': 'scikit-image',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
        'dateutil': 'python-dateutil',
    }
    
    all_imports = set()
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(project_root):
        # Skip hidden directories and cache
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    print(f"📁 Found {len(python_files)} Python files")
    print("-" * 50)
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Find import statements
                imports = re.findall(r'^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)', 
                                   content, re.MULTILINE)
                
                for imp in imports:
                    # Get base module name (before first dot)
                    base_module = imp.split('.')[0]
                    
                    if base_module and not base_module.startswith('_'):
                        # Check if it's standard library
                        if base_module in stdlib_modules:
                            continue
                        
                        # Convert to pip name if needed
                        pip_name = pip_name_map.get(base_module, base_module)
                        all_imports.add(pip_name)
                        
        except Exception as e:
            print(f"⚠️  Could not read {py_file}: {e}")
    
    return sorted(all_imports)

def get_current_versions():
    """Try to get current versions of installed packages"""
    version_info = {}
    
    version_checks = {
        'opencv-python': ('cv2', '__version__'),
        'numpy': ('numpy', '__version__'),
        'Pillow': ('PIL.Image', '__version__'),
        'matplotlib': ('matplotlib', '__version__'),
        'scipy': ('scipy', '__version__'),
        'scikit-image': ('skimage', '__version__'),
        'scikit-learn': ('sklearn', '__version__'),
    }
    
    for pip_name, (import_path, attr) in version_checks.items():
        try:
            # Dynamic import
            module = __import__(import_path.split('.')[0])
            
            # Navigate through dots if needed
            for part in import_path.split('.')[1:]:
                module = getattr(module, part)
            
            version = getattr(module, attr)
            version_info[pip_name] = version
        except:
            version_info[pip_name] = None
    
    return version_info

def main():
    print("🔍 SCANNING PROJECT FOR IMPORTS...")
    print()
    
    imports = find_imports()
    versions = get_current_versions()
    
    print("📦 THIRD-PARTY IMPORTS FOUND:")
    print("-" * 50)
    
    for imp in imports:
        version = versions.get(imp, "?")
        version_str = f"=={version}" if version else "==  # Add version"
        print(f"{imp}{version_str}")
    
    print()
    print("-" * 50)
    print("📝 SUGGESTED requirements.txt:")
    print("-" * 50)
    
    print("# Image Processing Project - Complete Requirements")
    print()
    for imp in imports:
        version = versions.get(imp, None)
        if version:
            print(f"{imp}=={version}")
        else:
            print(f"{imp}==  # Add appropriate version")
    
    print()
    print("# Optional development packages:")
    print("# pytest==7.4.3")
    print("# pytest-cov==4.1.0")
    
    print()
    print("💡 Copy the above to requirements.txt")
    print("💡 Then run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
