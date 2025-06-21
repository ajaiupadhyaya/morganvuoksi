#!/usr/bin/env python3
"""
MorganVuoksi Terminal Setup Verification
Checks that all components are properly installed and configured.
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TerminalVerifier:
    """Verifies MorganVuoksi Terminal setup."""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.issues = []
        self.warnings = []
        
    def run_verification(self):
        """Run complete verification."""
        print("🔍 MorganVuoksi Terminal Setup Verification")
        print("=" * 50)
        
        # Check Python version
        self.check_python_version()
        
        # Check required packages
        self.check_required_packages()
        
        # Check file structure
        self.check_file_structure()
        
        # Check configuration
        self.check_configuration()
        
        # Check API keys
        self.check_api_keys()
        
        # Check data sources
        self.check_data_sources()
        
        # Check models
        self.check_models()
        
        # Summary
        self.print_summary()
        
    def check_python_version(self):
        """Check Python version."""
        print("\n🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        else:
            self.issues.append(f"Python {version.major}.{version.minor}.{version.micro} is too old. Need 3.8+")
            print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Too old")
    
    def check_required_packages(self):
        """Check required packages."""
        print("\n📦 Checking required packages...")
        
        required_packages = [
            'streamlit',
            'pandas',
            'numpy',
            'plotly',
            'yfinance',
            'scikit-learn',
            'torch',
            'xgboost',
            'statsmodels',
            'arch',
            'transformers',
            'textblob',
            'nltk',
            'aiohttp',
            'requests',
            'scipy'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                importlib.import_module(package.replace('-', '_'))
                print(f"✅ {package} - OK")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package} - Missing")
        
        if missing_packages:
            self.issues.append(f"Missing packages: {', '.join(missing_packages)}")
            print(f"\n💡 Install missing packages with: pip install {' '.join(missing_packages)}")
    
    def check_file_structure(self):
        """Check file structure."""
        print("\n📁 Checking file structure...")
        
        required_files = [
            'dashboard/terminal.py',
            'src/data/market_data.py',
            'src/models/advanced_models.py',
            'src/models/rl_models.py',
            'src/signals/nlp_signals.py',
            'src/portfolio/optimizer.py',
            'src/risk/risk_manager.py',
            'src/visuals/dashboard.py',
            'requirements-dashboard.txt',
            'run_terminal.sh',
            'README.md'
        ]
        
        required_dirs = [
            'src',
            'dashboard',
            'src/data',
            'src/models',
            'src/signals',
            'src/portfolio',
            'src/risk',
            'src/visuals',
            'logs',
            'outputs',
            'models/saved_models'
        ]
        
        # Check directories
        for dir_path in required_dirs:
            full_path = self.root_dir / dir_path
            if full_path.exists():
                print(f"✅ {dir_path}/ - OK")
            else:
                self.warnings.append(f"Directory {dir_path} not found")
                print(f"⚠️  {dir_path}/ - Not found")
        
        # Check files
        for file_path in required_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                print(f"✅ {file_path} - OK")
            else:
                self.issues.append(f"File {file_path} not found")
                print(f"❌ {file_path} - Missing")
    
    def check_configuration(self):
        """Check configuration files."""
        print("\n⚙️  Checking configuration...")
        
        config_files = [
            'config/config.yaml',
            'config/base.yaml'
        ]
        
        for config_file in config_files:
            full_path = self.root_dir / config_file
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        yaml.safe_load(f)
                    print(f"✅ {config_file} - Valid YAML")
                except yaml.YAMLError as e:
                    self.issues.append(f"Invalid YAML in {config_file}: {e}")
                    print(f"❌ {config_file} - Invalid YAML")
            else:
                self.warnings.append(f"Configuration file {config_file} not found")
                print(f"⚠️  {config_file} - Not found")
    
    def check_api_keys(self):
        """Check API key configuration."""
        print("\n🔑 Checking API keys...")
        
        # Check .env file
        env_file = self.root_dir / '.env'
        if env_file.exists():
            print("✅ .env file found")
            
            # Read and check for common API keys
            with open(env_file, 'r') as f:
                content = f.read()
                
            api_keys = [
                'ALPACA_API_KEY',
                'POLYGON_API_KEY',
                'FRED_API_KEY',
                'OPENAI_API_KEY',
                'NEWS_API_KEY'
            ]
            
            found_keys = []
            for key in api_keys:
                if key in content:
                    found_keys.append(key)
            
            if found_keys:
                print(f"✅ Found API keys: {', '.join(found_keys)}")
            else:
                self.warnings.append("No API keys found in .env file")
                print("⚠️  No API keys found")
        else:
            self.warnings.append(".env file not found")
            print("⚠️  .env file not found")
        
        # Check environment variables
        env_vars = [
            'ALPACA_API_KEY',
            'POLYGON_API_KEY',
            'FRED_API_KEY',
            'OPENAI_API_KEY',
            'NEWS_API_KEY'
        ]
        
        found_env_vars = []
        for var in env_vars:
            if os.getenv(var):
                found_env_vars.append(var)
        
        if found_env_vars:
            print(f"✅ Environment variables: {', '.join(found_env_vars)}")
        else:
            print("ℹ️  No API keys in environment variables")
    
    def check_data_sources(self):
        """Check data source connectivity."""
        print("\n📊 Checking data sources...")
        
        try:
            import yfinance as yf
            # Test Yahoo Finance
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            if info:
                print("✅ Yahoo Finance - Connected")
            else:
                self.warnings.append("Yahoo Finance connection failed")
                print("⚠️  Yahoo Finance - Connection failed")
        except Exception as e:
            self.warnings.append(f"Yahoo Finance error: {e}")
            print(f"⚠️  Yahoo Finance - Error: {e}")
        
        # Check if Alpaca is configured
        if os.getenv('ALPACA_API_KEY'):
            print("✅ Alpaca API - Configured")
        else:
            print("ℹ️  Alpaca API - Not configured")
        
        # Check if Polygon is configured
        if os.getenv('POLYGON_API_KEY'):
            print("✅ Polygon API - Configured")
        else:
            print("ℹ️  Polygon API - Not configured")
    
    def check_models(self):
        """Check ML model availability."""
        print("\n🤖 Checking ML models...")
        
        try:
            import torch
            if torch.cuda.is_available():
                print("✅ PyTorch with CUDA - Available")
            else:
                print("✅ PyTorch CPU - Available")
        except ImportError:
            self.warnings.append("PyTorch not available")
            print("⚠️  PyTorch - Not available")
        
        try:
            import xgboost as xgb
            print("✅ XGBoost - Available")
        except ImportError:
            self.warnings.append("XGBoost not available")
            print("⚠️  XGBoost - Not available")
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print("✅ Transformers - Available")
        except ImportError:
            self.warnings.append("Transformers not available")
            print("⚠️  Transformers - Not available")
        
        try:
            import statsmodels.api as sm
            print("✅ Statsmodels - Available")
        except ImportError:
            self.warnings.append("Statsmodels not available")
            print("⚠️  Statsmodels - Not available")
    
    def print_summary(self):
        """Print verification summary."""
        print("\n" + "=" * 50)
        print("📋 VERIFICATION SUMMARY")
        print("=" * 50)
        
        if not self.issues and not self.warnings:
            print("🎉 All checks passed! MorganVuoksi Terminal is ready to use.")
            print("\n🚀 To launch the terminal:")
            print("   ./run_terminal.sh")
            print("   or")
            print("   cd dashboard && streamlit run terminal.py")
        else:
            if self.issues:
                print(f"\n❌ {len(self.issues)} critical issues found:")
                for issue in self.issues:
                    print(f"   • {issue}")
            
            if self.warnings:
                print(f"\n⚠️  {len(self.warnings)} warnings:")
                for warning in self.warnings:
                    print(f"   • {warning}")
            
            print("\n💡 Recommendations:")
            if self.issues:
                print("   • Fix critical issues before using the terminal")
            if self.warnings:
                print("   • Address warnings for optimal functionality")
            print("   • Install missing packages: pip install -r requirements-dashboard.txt")
            print("   • Configure API keys in .env file for full functionality")
        
        print("\n📚 For more information, see README.md")
        print("🆘 For support, check the documentation or create an issue")

def main():
    """Main verification function."""
    verifier = TerminalVerifier()
    verifier.run_verification()

if __name__ == "__main__":
    main() 