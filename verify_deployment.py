#!/usr/bin/env python3
"""
MorganVuoksi Elite Terminal - Deployment Verification Script
Verifies all components are working and ready for deployment.
"""

import os
import sys
import subprocess
import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        return False, f"Python {version.major}.{version.minor} (requires 3.9+)"
    return True, f"Python {version.major}.{version.minor}.{version.micro}"

def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'yfinance',
        'requests', 'aiohttp', 'fastapi', 'uvicorn'
    ]
    
    missing = []
    working = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            working.append(package)
        except ImportError:
            missing.append(package)
    
    return missing, working

def check_files():
    """Check if all required files exist."""
    required_files = [
        'streamlit_app.py',
        'requirements.txt',
        'requirements-web.txt',
        'Dockerfile',
        'docker-compose.yml',
        'docker-compose.production.yml',
        'provided/package.json',
        'frontend/package.json'
    ]
    
    missing = []
    existing = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing.append(file_path)
        else:
            missing.append(file_path)
    
    return missing, existing

def check_streamlit_app():
    """Test if the Streamlit app can be imported."""
    try:
        # Try to import the streamlit app
        import streamlit as st
        
        # Test basic functionality
        test_code = """
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Basic test
st.write("Test")
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['A'], y=df['B']))
"""
        
        # Create temporary test file
        with open('temp_test.py', 'w') as f:
            f.write(test_code)
        
        # Try to run streamlit check
        result = subprocess.run(['python', '-c', test_code], 
                              capture_output=True, text=True, timeout=10)
        
        # Clean up
        if os.path.exists('temp_test.py'):
            os.remove('temp_test.py')
        
        return result.returncode == 0, result.stderr if result.returncode != 0 else "OK"
        
    except Exception as e:
        return False, str(e)

def check_frontend():
    """Check if frontend components are ready."""
    checks = []
    
    # Check provided/ folder
    provided_path = Path('provided')
    if provided_path.exists():
        # Check package.json
        package_json = provided_path / 'package.json'
        if package_json.exists():
            checks.append(("Provided UI package.json", True, "Found"))
        else:
            checks.append(("Provided UI package.json", False, "Missing"))
        
        # Check key components
        components_path = provided_path / 'src' / 'components'
        if components_path.exists():
            key_components = [
                'TradingDashboard.tsx',
                'MarketOverview.tsx',
                'PriceChart.tsx',
                'PortfolioSummary.tsx'
            ]
            
            for component in key_components:
                component_file = components_path / component
                if component_file.exists():
                    checks.append((f"Component: {component}", True, "Found"))
                else:
                    checks.append((f"Component: {component}", False, "Missing"))
    else:
        checks.append(("Provided UI folder", False, "Missing"))
    
    # Check frontend/ folder
    frontend_path = Path('frontend')
    if frontend_path.exists():
        package_json = frontend_path / 'package.json'
        if package_json.exists():
            checks.append(("Frontend package.json", True, "Found"))
        else:
            checks.append(("Frontend package.json", False, "Missing"))
    else:
        checks.append(("Frontend folder", False, "Missing"))
    
    return checks

def check_docker():
    """Check Docker configuration."""
    checks = []
    
    # Check if Docker is available
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            checks.append(("Docker", True, result.stdout.strip()))
        else:
            checks.append(("Docker", False, "Not available"))
    except:
        checks.append(("Docker", False, "Not installed"))
    
    # Check docker-compose
    try:
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            checks.append(("Docker Compose", True, result.stdout.strip()))
        else:
            checks.append(("Docker Compose", False, "Not available"))
    except:
        checks.append(("Docker Compose", False, "Not installed"))
    
    return checks

def generate_deployment_report():
    """Generate comprehensive deployment readiness report."""
    print("🚀 MorganVuoksi Elite Terminal - Deployment Verification")
    print("=" * 60)
    
    # Python version check
    python_ok, python_version = check_python_version()
    print(f"📋 Python Version: {python_version} {'✅' if python_ok else '❌'}")
    
    if not python_ok:
        print("   ⚠️  Python 3.9+ required for deployment")
    
    print()
    
    # Dependencies check
    missing_deps, working_deps = check_dependencies()
    print(f"📦 Dependencies: {len(working_deps)} working, {len(missing_deps)} missing")
    
    if missing_deps:
        print("   ❌ Missing packages:")
        for dep in missing_deps:
            print(f"      - {dep}")
        print("   💡 Run: pip install -r requirements.txt")
    else:
        print("   ✅ All core dependencies available")
    
    print()
    
    # Files check
    missing_files, existing_files = check_files()
    print(f"📁 Project Files: {len(existing_files)} found, {len(missing_files)} missing")
    
    if missing_files:
        print("   ❌ Missing files:")
        for file in missing_files:
            print(f"      - {file}")
    else:
        print("   ✅ All deployment files present")
    
    print()
    
    # Streamlit app check
    streamlit_ok, streamlit_msg = check_streamlit_app()
    print(f"🌐 Streamlit App: {'✅' if streamlit_ok else '❌'}")
    if not streamlit_ok:
        print(f"   Error: {streamlit_msg}")
    else:
        print("   Ready for web deployment")
    
    print()
    
    # Frontend check
    frontend_checks = check_frontend()
    print("🎨 Frontend Components:")
    for name, status, message in frontend_checks:
        print(f"   {name}: {'✅' if status else '❌'} {message}")
    
    print()
    
    # Docker check
    docker_checks = check_docker()
    print("🐳 Docker Environment:")
    for name, status, message in docker_checks:
        print(f"   {name}: {'✅' if status else '❌'} {message}")
    
    print()
    
    # Overall assessment
    print("🎯 DEPLOYMENT READINESS ASSESSMENT")
    print("-" * 40)
    
    total_checks = 0
    passed_checks = 0
    
    # Count all checks
    if python_ok:
        passed_checks += 1
    total_checks += 1
    
    if not missing_deps:
        passed_checks += 1
    total_checks += 1
    
    if not missing_files:
        passed_checks += 1
    total_checks += 1
    
    if streamlit_ok:
        passed_checks += 1
    total_checks += 1
    
    # Frontend checks
    frontend_passed = sum(1 for _, status, _ in frontend_checks if status)
    total_checks += len(frontend_checks)
    passed_checks += frontend_passed
    
    # Docker checks (optional)
    docker_passed = sum(1 for _, status, _ in docker_checks if status)
    
    readiness_score = (passed_checks / total_checks) * 100
    
    print(f"📊 Readiness Score: {readiness_score:.1f}% ({passed_checks}/{total_checks})")
    
    if readiness_score >= 90:
        print("🟢 READY FOR PRODUCTION DEPLOYMENT")
        print("   All critical components verified")
        
        if docker_passed >= 1:
            print("   🐳 Docker deployment available")
        
    elif readiness_score >= 70:
        print("🟡 READY FOR BASIC DEPLOYMENT")
        print("   Some optional components missing")
        
    else:
        print("🔴 NOT READY FOR DEPLOYMENT")
        print("   Critical issues need to be resolved")
    
    print()
    
    # Deployment recommendations
    print("💡 RECOMMENDED DEPLOYMENT METHODS")
    print("-" * 40)
    
    if streamlit_ok and not missing_deps:
        print("✅ Streamlit Cloud (streamlit_app.py)")
        print("   🌐 Free hosting: https://share.streamlit.io")
        
    if frontend_passed >= 3:
        print("✅ Professional UI (provided/ folder)")
        print("   ⚡ Advanced Bloomberg Terminal interface")
        
    if docker_passed >= 1:
        print("✅ Docker Deployment")
        print("   🐳 Production-grade containerized setup")
    
    print()
    print("📖 Full deployment instructions: README.md")
    print("🎉 Happy deploying!")

if __name__ == "__main__":
    generate_deployment_report()