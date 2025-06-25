#!/usr/bin/env python3
"""
MorganVuoksi Terminal - Startup Script
Handles dependency checks and graceful fallbacks for missing modules.
"""

import sys
import os
import warnings
from typing import Dict, List, Optional

# Suppress warnings
warnings.filterwarnings('ignore')

def check_dependencies() -> Dict[str, bool]:
    """Check which dependencies are available."""
    dependencies = {
        'streamlit': False,
        'pandas': False,
        'numpy': False,
        'plotly': False,
        'yfinance': False,
        'scikit-learn': False,
        'xgboost': False,
        'transformers': False,
        'statsmodels': False,
        'arch': False,
        'cvxpy': False,
        'alpaca-py': False,
        'polygon-api-client': False,
        'textblob': False,
        'nltk': False,
        'beautifulsoup4': False,
        'requests': False,
        'aiohttp': False,
        'websockets': False,
        'matplotlib': False,
        'seaborn': False,
        'altair': False,
        'redis': False,
        'sqlalchemy': False,
        'joblib': False,
        'tqdm': False,
        'pyyaml': False,
    }
    
    for dep in dependencies.keys():
        try:
            __import__(dep.replace('-', '_'))
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies

def check_core_modules() -> Dict[str, bool]:
    """Check if core application modules are available."""
    modules = {
        'src.data.market_data': False,
        'src.models.advanced_models': False,
        'src.models.rl_models': False,
        'src.signals.nlp_signals': False,
        'src.portfolio.optimizer': False,
        'src.risk.risk_manager': False,
        'src.visuals.charting': False,
    }
    
    # Add project root to path
    project_root = os.path.join(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    for module in modules.keys():
        try:
            __import__(module)
            modules[module] = True
        except ImportError:
            modules[module] = False
    
    return modules

def print_status_report():
    """Print a status report of available dependencies and modules."""
    print("🔍 MorganVuoksi Terminal - Dependency Check")
    print("=" * 50)
    
    # Check dependencies
    deps = check_dependencies()
    print("\n📦 Dependencies:")
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"  {status} {dep}")
    
    # Check modules
    modules = check_core_modules()
    print("\n📁 Core Modules:")
    for module, available in modules.items():
        status = "✅" if available else "❌"
        print(f"  {status} {module}")
    
    # Summary
    core_deps = ['streamlit', 'pandas', 'numpy', 'plotly', 'yfinance']
    core_available = all(deps[dep] for dep in core_deps)
    
    print("\n📊 Summary:")
    if core_available:
        print("✅ Core functionality available - Terminal will run")
    else:
        print("❌ Core dependencies missing - Terminal may not function properly")
    
    missing_deps = [dep for dep, available in deps.items() if not available]
    if missing_deps:
        print(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
    
    missing_modules = [module for module, available in modules.items() if not available]
    if missing_modules:
        print(f"⚠️ Missing modules: {', '.join(missing_modules)}")

def main():
    """Main startup function."""
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        print_status_report()
        return
    
    # Check if we can run the terminal
    deps = check_dependencies()
    core_deps = ['streamlit', 'pandas', 'numpy', 'plotly', 'yfinance']
    core_available = all(deps[dep] for dep in core_deps)
    
    if not core_available:
        print("❌ Critical dependencies missing. Please install required packages.")
        print("Run: pip install streamlit pandas numpy plotly yfinance")
        sys.exit(1)
    
    # If we get here, we can run the terminal
    print("✅ All core dependencies available. Starting MorganVuoksi Terminal...")
    
    # Import and run the terminal
    try:
        import streamlit as st
        from dashboard.terminal import main as terminal_main
        terminal_main()
    except Exception as e:
        print(f"❌ Error starting terminal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 