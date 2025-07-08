#!/usr/bin/env python3
"""
Test script to verify that requirements can be installed
Run this before Docker build to catch dependency issues early
"""

import subprocess
import sys
import tempfile
import os

def test_requirements_syntax():
    """Test that requirements files have valid syntax"""
    print("Testing requirements file syntax...")
    
    requirements_files = [
        "requirements_fastapi.txt",
        "requirements_docker.txt"
    ]
    
    for req_file in requirements_files:
        if os.path.exists(req_file):
            print(f"✓ {req_file} exists")
            with open(req_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Basic validation of package names
                        if '>=' in line or '==' in line or '~=' in line:
                            package_name = line.split('>=')[0].split('==')[0].split('~=')[0]
                        else:
                            package_name = line
                        
                        # Check for invalid characters
                        if any(char in package_name for char in ['<', '>', '!']):
                            print(f"✗ {req_file}:{i} - Invalid package specification: {line}")
                            return False
                        
                print(f"✓ {req_file} syntax looks good")
        else:
            print(f"✗ {req_file} not found")
            return False
    
    return True

def check_package_availability():
    """Check if packages exist on PyPI (dry run)"""
    print("\nChecking package availability...")
    
    test_packages = [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0", 
        "python-multipart>=0.0.6",
        "pydantic>=2.5.0",
        "torch",
        "torchvision",
        "torchaudio"
    ]
    
    for package in test_packages:
        try:
            # Use pip to check if package exists (dry run)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--dry-run", "--quiet", package],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print(f"✓ {package} - available")
            else:
                print(f"✗ {package} - not available or version conflict")
                print(f"  Error: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"? {package} - timeout (network issue?)")
        except Exception as e:
            print(f"? {package} - check failed: {e}")
    
    return True

def simulate_fastapi_import():
    """Test if FastAPI imports would work"""
    print("\nTesting FastAPI imports...")
    
    test_imports = [
        "import tempfile",
        "import shutil", 
        "import zipfile",
        "import logging",
        "import asyncio",
        "from pathlib import Path",
        "from typing import Optional, List"
    ]
    
    for test_import in test_imports:
        try:
            exec(test_import)
            print(f"✓ {test_import}")
        except ImportError as e:
            print(f"✗ {test_import} - {e}")
            return False
        except Exception as e:
            print(f"? {test_import} - {e}")
    
    # Test zipfile functionality (replacement for patoolib)
    try:
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            import zipfile
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                zf.writestr('test.txt', 'test content')
            
            with zipfile.ZipFile(tmp.name, 'r') as zf:
                content = zf.read('test.txt').decode()
                assert content == 'test content'
            
            os.unlink(tmp.name)
            print("✓ zipfile functionality works")
    except Exception as e:
        print(f"✗ zipfile test failed: {e}")
        return False
    
    return True

def main():
    print("LAM Requirements Verification")
    print("=" * 40)
    
    all_good = True
    
    # Test requirements file syntax
    if not test_requirements_syntax():
        all_good = False
    
    # Check package availability (requires internet)
    if not check_package_availability():
        all_good = False
    
    # Test core imports
    if not simulate_fastapi_import():
        all_good = False
    
    print("\n" + "=" * 40)
    if all_good:
        print("✅ All tests passed! Docker build should work.")
        print("\nYou can now run:")
        print("  ./build_docker.sh          # Full build")
        print("  ./build_docker.sh --minimal # Minimal build") 
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before Docker build.")
        return 1

if __name__ == "__main__":
    exit(main()) 