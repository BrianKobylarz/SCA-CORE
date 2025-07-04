#!/usr/bin/env python3
"""Test script to verify the package structure without importing heavy dependencies."""

import sys
import importlib.util
from pathlib import Path

def check_file_exists(file_path):
    """Check if a file exists."""
    path = Path(file_path)
    exists = path.exists()
    print(f"  {'✅' if exists else '❌'} {file_path}")
    return exists

def check_module_syntax(module_path):
    """Check if a Python module has valid syntax."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        module = importlib.util.module_from_spec(spec)
        # Don't execute, just check syntax
        with open(module_path, 'r') as f:
            compile(f.read(), module_path, 'exec')
        return True
    except SyntaxError as e:
        print(f"    Syntax error: {e}")
        return False
    except Exception as e:
        print(f"    Other error: {e}")
        return False

def main():
    print("🔍 Checking Semantic Flow Analyzer package structure...")
    print()
    
    base_path = Path("sfa")
    
    # Check critical files we created
    critical_files = [
        "sfa/__init__.py",
        "sfa/core/__init__.py", 
        "sfa/core/base.py",
        "sfa/core/types.py",
        "sfa/core/embeddings.py",
        "sfa/core/flow_manager.py",
        "sfa/config/__init__.py",
        "sfa/config/flow_config.py",
        "sfa/config/visualization_config.py",
        "sfa/io/__init__.py",
        "sfa/io/reddit_loader.py",
        "sfa/io/cache_manager.py",
        "sfa/sparse/__init__.py",
        "sfa/sparse/interpolation.py",
        "sfa/sparse/confidence_bands.py",
        "sfa/visualization/__init__.py",
        "sfa/visualization/layer_manager.py"
    ]
    
    print("📁 Critical Files:")
    all_exist = True
    for file_path in critical_files:
        exists = check_file_exists(file_path)
        if not exists:
            all_exist = False
    
    print()
    print("🔧 Syntax Check:")
    syntax_ok = True
    for file_path in critical_files:
        if file_path.endswith('.py') and Path(file_path).exists():
            print(f"  Checking {file_path}...")
            if not check_module_syntax(file_path):
                syntax_ok = False
    
    print()
    print("📊 Summary:")
    print(f"  Files exist: {'✅' if all_exist else '❌'}")
    print(f"  Syntax valid: {'✅' if syntax_ok else '❌'}")
    
    if all_exist and syntax_ok:
        print()
        print("🎉 Package structure is complete and ready!")
        print()
        print("📋 Next steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Install package: pip install -e .")
        print("  3. Run examples: python examples/complete_analysis_example.py")
        return True
    else:
        print()
        print("❌ Package has issues that need to be fixed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)