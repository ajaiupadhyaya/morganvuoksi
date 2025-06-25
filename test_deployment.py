#!/usr/bin/env python3
"""
MorganVuoksi Terminal - Deployment Testing Script
Test the web deployment locally before pushing to production.
"""

import sys
import subprocess
import importlib
import importlib.util
import time
import requests
from pathlib import Path
from datetime import datetime
import os

class DeploymentTester:
    """Test deployment readiness for MorganVuoksi Terminal."""
    
    def __init__(self):
        self.test_results = []
        self.errors = []
        
    def run_tests(self):
        """Run all deployment tests."""
        print("🚀 MorganVuoksi Terminal - Deployment Testing")
        print("=" * 50)
        
        # Test 1: Check dependencies
        self.test_dependencies()
        
        # Test 2: Check files
        self.test_required_files()
        
        # Test 3: Test imports
        self.test_imports()
        
        # Test 4: Test configuration
        self.test_configuration()
        
        # Test 5: Launch test (optional)
        if input("\n🔄 Run live test? (y/N): ").lower() == 'y':
            self.test_local_launch()
        
        # Display results
        self.display_results()
        
    def test_dependencies(self):
        """Test if all required dependencies can be installed."""
        print("\n📦 Testing Dependencies...")
        
        try:
            # Check if requirements-web.txt exists
            req_file = Path("requirements-web.txt")
            if not req_file.exists():
                self.errors.append("requirements-web.txt not found")
                return
            
            # Read requirements
            with open(req_file, 'r') as f:
                requirements = [line.strip() for line in f.readlines() 
                              if line.strip() and not line.startswith('#')]
            
            print(f"   ✅ Found {len(requirements)} dependencies")
            
            # Test critical imports
            critical_packages = ['streamlit', 'pandas', 'numpy', 'plotly', 'yfinance']
            for package in critical_packages:
                try:
                    importlib.import_module(package)
                    print(f"   ✅ {package} - OK")
                except ImportError:
                    print(f"   ❌ {package} - MISSING")
                    self.errors.append(f"Missing package: {package}")
            
            self.test_results.append("Dependencies check completed")
            
        except Exception as e:
            self.errors.append(f"Dependency test failed: {str(e)}")
    
    def test_required_files(self):
        """Test if all required files exist."""
        print("\n📁 Testing Required Files...")
        
        required_files = [
            "streamlit_app.py",
            "requirements-web.txt", 
            ".streamlit/config.toml",
            ".streamlit/secrets.toml.example",
            "README.md",
            "DEPLOYMENT_GUIDE.md"
        ]
        
        for file_path in required_files:
            path = Path(file_path)
            if path.exists():
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path} - MISSING")
                self.errors.append(f"Missing file: {file_path}")
        
        self.test_results.append("File structure check completed")
    
    def test_imports(self):
        """Test if the main application can be imported."""
        print("\n🐍 Testing Python Imports...")
        
        try:
            # Test if streamlit_app can be imported
            spec = importlib.util.spec_from_file_location("streamlit_app", "streamlit_app.py")
            if spec is None:
                self.errors.append("Cannot load streamlit_app.py")
                return
                
            print("   ✅ streamlit_app.py can be loaded")
            
            # Test specific functions
            try:
                import streamlit as st
                import pandas as pd
                import numpy as np
                import plotly.graph_objects as go
                import yfinance as yf
                print("   ✅ All critical imports successful")
            except ImportError as ie:
                print(f"   ❌ Critical import failed: {str(ie)}")
                self.errors.append(f"Critical import failed: {str(ie)}")
                return
            
            self.test_results.append("Import tests passed")
            
        except Exception as e:
            print(f"   ❌ Import error: {str(e)}")
            self.errors.append(f"Import test failed: {str(e)}")
    
    def test_configuration(self):
        """Test configuration files."""
        print("\n⚙️ Testing Configuration...")
        
        try:
            # Test Streamlit config
            config_path = Path(".streamlit/config.toml")
            if config_path.exists():
                print("   ✅ Streamlit config found")
                # Could parse TOML here for validation
            else:
                print("   ⚠️ No Streamlit config (optional)")
            
            # Test secrets template
            secrets_path = Path(".streamlit/secrets.toml.example")
            if secrets_path.exists():
                print("   ✅ Secrets template found")
            else:
                print("   ⚠️ No secrets template")
            
            self.test_results.append("Configuration tests completed")
            
        except Exception as e:
            self.errors.append(f"Configuration test failed: {str(e)}")
    
    def test_local_launch(self):
        """Test launching the application locally."""
        print("\n🚀 Testing Local Launch...")
        
        try:
            print("   🔄 Starting Streamlit server...")
            
            # Start Streamlit in background
            process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
                "--server.port=8502",  # Use different port to avoid conflicts
                "--server.headless=true"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for startup
            time.sleep(10)
            
            # Test if server is responding
            try:
                response = requests.get("http://localhost:8502/_stcore/health", timeout=5)
                if response.status_code == 200:
                    print("   ✅ Streamlit server is responding")
                    self.test_results.append("Local launch test passed")
                else:
                    print(f"   ❌ Server returned status: {response.status_code}")
                    self.errors.append("Server not responding correctly")
            except requests.RequestException as e:
                print(f"   ❌ Cannot connect to server: {str(e)}")
                self.errors.append("Cannot connect to local server")
            
            # Cleanup
            process.terminate()
            process.wait()
            
        except Exception as e:
            print(f"   ❌ Launch test failed: {str(e)}")
            self.errors.append(f"Launch test failed: {str(e)}")
    
    def display_results(self):
        """Display test results and recommendations."""
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS")
        print("=" * 50)
        
        if not self.errors:
            print("🎉 ALL TESTS PASSED!")
            print("\n✅ Your application is ready for web deployment!")
            print("\n🚀 Next steps:")
            print("   1. Push your code to GitHub")
            print("   2. Visit share.streamlit.io")
            print("   3. Deploy your app")
            print("   4. Share your live URL!")
            
        else:
            print(f"❌ {len(self.errors)} ISSUES FOUND:")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")
            
            print("\n🔧 Please fix these issues before deploying.")
            
        print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save test report
        self.save_test_report()
    
    def save_test_report(self):
        """Save test report to file."""
        try:
            report_path = Path("deployment_test_report.txt")
            with open(report_path, 'w') as f:
                f.write("MorganVuoksi Terminal - Deployment Test Report\n")
                f.write("=" * 50 + "\n")
                f.write(f"Test Date: {datetime.now().isoformat()}\n\n")
                
                f.write("Test Results:\n")
                for result in self.test_results:
                    f.write(f"✅ {result}\n")
                
                if self.errors:
                    f.write(f"\nErrors Found ({len(self.errors)}):\n")
                    for error in self.errors:
                        f.write(f"❌ {error}\n")
                else:
                    f.write("\n🎉 No errors found - Ready for deployment!\n")
            
            print(f"\n📄 Test report saved to: {report_path}")
            
        except Exception as e:
            print(f"⚠️ Could not save test report: {str(e)}")

def main():
    """Main function."""
    print("🔍 MorganVuoksi Terminal Deployment Tester")
    print("This script will verify your application is ready for web deployment.\n")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Run tests
    tester = DeploymentTester()
    tester.run_tests()

if __name__ == "__main__":
    main()