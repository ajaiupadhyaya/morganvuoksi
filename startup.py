#!/usr/bin/env python3
"""
MorganVuoksi Terminal - Railway.app Optimized Startup Script
Handles dependency checks, memory optimization, and graceful fallbacks.
"""

import sys
import os
import warnings
import gc
import psutil
from typing import Dict, List, Optional

# Suppress warnings
warnings.filterwarnings('ignore')

def get_system_info() -> Dict[str, str]:
    """Get system information for debugging."""
    try:
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'memory_total': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            'memory_available': f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
            'cpu_count': str(psutil.cpu_count()),
            'railway_port': os.getenv('PORT', 'Not set'),
            'streamlit_port': os.getenv('STREAMLIT_SERVER_PORT', 'Not set')
        }
    except Exception:
        return {'system_info': 'Unable to retrieve'}

def check_dependencies() -> Dict[str, bool]:
    """Check which dependencies are available with Railway optimizations."""
    dependencies = {
        # Core requirements
        'streamlit': False,
        'pandas': False,
        'numpy': False,
        'plotly': False,
        'yfinance': False,
        
        # ML libraries (CPU optimized)
        'sklearn': False,
        'xgboost': False,
        'torch': False,
        'transformers': False,
        'statsmodels': False,
        'arch': False,
        
        # Financial APIs
        'alpaca': False,  # alpaca-py
        'polygon': False,
        
        # Utilities
        'requests': False,
        'aiohttp': False,
        'websockets': False,
        'redis': False,
        'sqlalchemy': False,
        'joblib': False,
        'tqdm': False,
        'yaml': False,
    }
    
    # Check each dependency
    for dep in dependencies.keys():
        try:
            if dep == 'sklearn':
                import sklearn
            elif dep == 'alpaca':
                from alpaca.trading import TradingClient
            elif dep == 'polygon':
                from polygon import RESTClient
            elif dep == 'yaml':
                import yaml
            else:
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
        'dashboard.terminal': False,
        'src.api.main': False,
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

def optimize_memory():
    """Optimize memory usage for Railway deployment."""
    try:
        # Force garbage collection
        gc.collect()
        
        # Set memory-efficient settings for PyTorch if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Use CPU-only mode to save memory
            torch.set_num_threads(min(4, psutil.cpu_count()))
        except ImportError:
            pass
        
        # Set environment variables for memory optimization
        os.environ['OMP_NUM_THREADS'] = str(min(4, psutil.cpu_count()))
        os.environ['MKL_NUM_THREADS'] = str(min(4, psutil.cpu_count()))
        os.environ['NUMEXPR_NUM_THREADS'] = str(min(4, psutil.cpu_count()))
        
        print("✅ Memory optimization applied")
        
    except Exception as e:
        print(f"⚠️ Memory optimization failed: {e}")

def print_status_report():
    """Print a comprehensive status report for Railway deployment."""
    print("🚀 MorganVuoksi Terminal - Railway.app Deployment Check")
    print("=" * 60)
    
    # System information
    sys_info = get_system_info()
    print("\n🖥️ System Information:")
    for key, value in sys_info.items():
        print(f"  • {key}: {value}")
    
    # Check dependencies
    deps = check_dependencies()
    print("\n📦 Dependencies Status:")
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"  {status} {dep}")
    
    # Check modules
    modules = check_core_modules()
    print("\n📁 Core Modules Status:")
    for module, available in modules.items():
        status = "✅" if available else "❌"
        print(f"  {status} {module}")
    
    # Environment variables check
    print("\n🔧 Environment Variables:")
    env_vars = [
        'PORT', 'STREAMLIT_SERVER_PORT', 'STREAMLIT_SERVER_ADDRESS',
        'ALPACA_API_KEY', 'POLYGON_API_KEY', 'OPENAI_API_KEY'
    ]
    for var in env_vars:
        value = os.getenv(var)
        status = "✅" if value else "❌"
        display_value = "***" if value and "key" in var.lower() else (value or "Not set")
        print(f"  {status} {var}: {display_value}")
    
    # Summary
    core_deps = ['streamlit', 'pandas', 'numpy', 'plotly', 'yfinance']
    core_available = all(deps[dep] for dep in core_deps)
    
    print("\n📊 Railway Deployment Summary:")
    if core_available:
        print("✅ Core functionality available - Terminal ready for Railway")
    else:
        print("❌ Core dependencies missing - Deployment will fail")
    
    # Performance recommendations
    memory = psutil.virtual_memory()
    if memory.available < 1 * 1024**3:  # Less than 1GB
        print("⚠️ Low memory detected - Consider upgrading Railway plan")
    
    missing_deps = [dep for dep, available in deps.items() if not available]
    if missing_deps:
        print(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
    
    missing_modules = [module for module, available in modules.items() if not available]
    if missing_modules:
        print(f"⚠️ Missing modules: {', '.join(missing_modules)}")

def setup_railway_environment():
    """Setup Railway-specific environment configurations."""
    # Ensure PORT is properly configured
    port = os.getenv('PORT', '8501')
    os.environ['STREAMLIT_SERVER_PORT'] = port
    
    # Railway-specific Streamlit config
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    
    # Production settings
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
    
    print(f"✅ Railway environment configured (Port: {port})")

def main():
    """Main startup function optimized for Railway."""
    print("🔄 Starting MorganVuoksi Terminal...")
    
    # Check if we're just running a dependency check
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        print_status_report()
        return
    
    # Apply memory optimizations
    optimize_memory()
    
    # Setup Railway environment
    setup_railway_environment()
    
    # Check core dependencies
    deps = check_dependencies()
    core_deps = ['streamlit', 'pandas', 'numpy', 'plotly']
    core_available = all(deps[dep] for dep in core_deps)
    
    if not core_available:
        print("❌ Critical dependencies missing. Railway deployment will fail.")
        print("Missing core packages. Check Railway build logs.")
        sys.exit(1)
    
    # Check if terminal module is available
    try:
        from dashboard.terminal import main as terminal_main
        print("✅ Terminal module loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to import terminal module: {e}")
        print("Attempting fallback startup...")
        try:
            import streamlit as st
            st.error("Terminal module not available. Check deployment.")
            st.stop()
        except:
            sys.exit(1)
    
    # Start the terminal with error handling
    try:
        print(f"🚀 Starting MorganVuoksi Terminal on port {os.getenv('PORT', '8501')}")
        print("✅ All systems operational - Railway deployment successful!")
        
        # Start terminal
        terminal_main()
        
    except Exception as e:
        print(f"❌ Error starting terminal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 