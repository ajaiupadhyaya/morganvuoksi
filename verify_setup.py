#!/usr/bin/env python3
"""
MorganVuoksi Terminal Setup Verification
This script verifies that all dependencies and configuration are properly set up.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit', 'plotly', 'pandas', 'numpy', 'yfinance',
        'scikit-learn', 'xgboost', 'tensorflow', 'torch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Installed")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements-dashboard.txt")
        return False
    
    return True

def check_files():
    """Check if required files exist."""
    required_files = [
        'dashboard/terminal.py',
        'config/config.yaml',
        'requirements-dashboard.txt',
        'run_terminal.sh'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} - Found")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n📁 Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def check_env_file():
    """Check if .env file exists and has required variables."""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("Create .env file with API keys (see README.md for details)")
        return True  # Not critical for basic functionality
    
    print("✅ .env file - Found")
    
    # Check for common API keys
    with open(env_file, 'r') as f:
        content = f.read()
        
    api_keys = {
        'OPENAI_API_KEY': 'OpenAI GPT (Optional)',
        'ALPACA_API_KEY': 'Alpaca Trading (Optional)',
        'POLYGON_API_KEY': 'Polygon.io (Optional)'
    }
    
    for key, description in api_keys.items():
        if key in content:
            print(f"✅ {key} - Configured ({description})")
        else:
            print(f"⚠️  {key} - Not configured ({description})")
    
    return True

def check_streamlit():
    """Check if Streamlit can be launched."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'streamlit', '--version'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Streamlit - {version}")
            return True
        else:
            print("❌ Streamlit - Failed to run")
            return False
    except Exception as e:
        print(f"❌ Streamlit - Error: {e}")
        return False

def check_data_access():
    """Check if market data can be accessed."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        if 'regularMarketPrice' in info and info['regularMarketPrice']:
            print("✅ Market Data - Yahoo Finance accessible")
            return True
        else:
            print("⚠️  Market Data - Limited access")
            return True
    except Exception as e:
        print(f"❌ Market Data - Error: {e}")
        return False

def main():
    """Run all verification checks."""
    print("🔍 MorganVuoksi Terminal Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Required Files", check_files),
        ("Environment", check_env_file),
        ("Streamlit", check_streamlit),
        ("Data Access", check_data_access)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n📋 Checking {check_name}...")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} - Error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Verification Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Setup verification successful!")
        print("You can now run the terminal with:")
        print("  streamlit run dashboard/terminal.py")
    else:
        print("\n⚠️  Some checks failed. Please review the issues above.")
        print("Refer to README.md for detailed setup instructions.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 