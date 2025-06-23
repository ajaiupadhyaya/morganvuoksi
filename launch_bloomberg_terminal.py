#!/usr/bin/env python3
"""
MorganVuoksi Elite Terminal Launcher
One-click Bloomberg-style terminal launcher with automatic setup.
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path
import webbrowser
from datetime import datetime

class TerminalLauncher:
    """Bloomberg-style terminal launcher with automatic setup."""
    
    def __init__(self):
        self.processes = []
        self.terminal_url = "http://localhost:8501"
        self.api_url = "http://localhost:8000"
        self.is_running = False
        
    def print_banner(self):
        """Print startup banner."""
        print("\n" + "="*80)
        print("""
 ███╗   ███╗ ██████╗ ██████╗  ██████╗  █████╗ ███╗   ██╗██╗   ██╗██╗   ██╗ ██████╗ ██╗  ██╗███████╗██╗
 ████╗ ████║██╔═══██╗██╔══██╗██╔════╝ ██╔══██╗████╗  ██║██║   ██║██║   ██║██╔═══██╗██║ ██╔╝██╔════╝██║
 ██╔████╔██║██║   ██║██████╔╝██║  ███╗███████║██╔██╗ ██║██║   ██║██║   ██║██║   ██║█████╔╝ ███████╗██║
 ██║╚██╔╝██║██║   ██║██╔══██╗██║   ██║██╔══██║██║╚██╗██║╚██╗ ██╔╝██║   ██║██║   ██║██╔═██╗ ╚════██║██║
 ██║ ╚═╝ ██║╚██████╔╝██║  ██║╚██████╔╝██║  ██║██║ ╚████║ ╚████╔╝ ╚██████╔╝╚██████╔╝██║  ██╗███████║██║
 ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═══╝   ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝
        """)
        print("                    🏆 ELITE TERMINAL - Bloomberg-Grade Quantitative Finance 🏆")
        print("="*80)
        print(f"⚡ Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("📈 Features: Real-time Data | AI Predictions | Portfolio Optimization | Risk Management")
        print("🤖 Models: LSTM | Transformers | XGBoost | RL Agents | FinBERT | GPT Assistant")
        print("="*80 + "\n")
    
    def check_dependencies(self):
        """Check and install required dependencies."""
        print("🔧 Checking dependencies...")
        
        required_packages = [
            'streamlit',
            'pandas',
            'numpy',
            'plotly',
            'requests',
            'fastapi',
            'uvicorn'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} (missing)")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️  Installing missing packages: {', '.join(missing_packages)}")
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', *missing_packages
                ], check=True, capture_output=True)
                print("✅ All dependencies installed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install dependencies: {e}")
                print("Please install manually: pip install streamlit pandas numpy plotly requests fastapi uvicorn")
                return False
        
        print("✅ All dependencies satisfied!\n")
        return True
    
    def start_api_server(self):
        """Start FastAPI backend server."""
        print("🚀 Starting API server...")
        
        try:
            # Check if API server file exists
            api_file = Path("src/api/main.py")
            if not api_file.exists():
                print("⚠️  API server not found, terminal will run in mock mode")
                return True
            
            # Start API server
            process = subprocess.Popen([
                sys.executable, '-m', 'uvicorn', 
                'src.api.main:app',
                '--host', '0.0.0.0',
                '--port', '8000',
                '--reload'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(('API Server', process))
            
            # Wait for server to start
            time.sleep(3)
            
            # Check if server is running
            if process.poll() is None:
                print("✅ API server started on http://localhost:8000")
                return True
            else:
                print("⚠️  API server failed to start, terminal will run in mock mode")
                return True
                
        except Exception as e:
            print(f"⚠️  Could not start API server: {e}")
            print("   Terminal will run in mock mode")
            return True
    
    def start_terminal(self):
        """Start Streamlit terminal."""
        print("🖥️  Starting Bloomberg Terminal...")
        
        try:
            # Check if terminal file exists
            terminal_file = Path("terminal_elite.py")
            if not terminal_file.exists():
                print("❌ Terminal file not found: terminal_elite.py")
                return False
            
            # Start Streamlit terminal
            process = subprocess.Popen([
                sys.executable, '-m', 'streamlit', 'run',
                'terminal_elite.py',
                '--server.port', '8501',
                '--server.address', '0.0.0.0',
                '--server.headless', 'true',
                '--browser.gatherUsageStats', 'false'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(('Terminal', process))
            
            # Wait for terminal to start
            time.sleep(5)
            
            # Check if terminal is running
            if process.poll() is None:
                print("✅ Bloomberg Terminal started on http://localhost:8501")
                return True
            else:
                print("❌ Terminal failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Could not start terminal: {e}")
            return False
    
    def open_browser(self):
        """Open terminal in browser."""
        print("🌐 Opening terminal in browser...")
        
        try:
            time.sleep(2)  # Give terminal time to fully load
            webbrowser.open(self.terminal_url)
            print("✅ Browser opened")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print(f"   Please open manually: {self.terminal_url}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, shutting down...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def monitor_processes(self):
        """Monitor running processes."""
        while self.is_running:
            try:
                for name, process in self.processes:
                    if process.poll() is not None:
                        print(f"⚠️  {name} has stopped unexpectedly")
                        # Attempt to restart critical processes
                        if name == 'Terminal':
                            print("🔄 Attempting to restart terminal...")
                            self.start_terminal()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"⚠️  Process monitor error: {e}")
                break
    
    def print_status(self):
        """Print system status."""
        print("\n" + "="*60)
        print("📊 SYSTEM STATUS")
        print("="*60)
        print(f"🖥️  Terminal:     {self.terminal_url}")
        print(f"⚡ API Server:   {self.api_url}")
        print(f"📈 Status:      {'🟢 RUNNING' if self.is_running else '🔴 STOPPED'}")
        print(f"🕐 Time:        {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        print("\n💡 QUICK ACTIONS:")
        print("   • Press Ctrl+C to shutdown")
        print("   • Open new terminal tab for other commands")
        print("   • Visit http://localhost:8501 for terminal")
        print("   • Visit http://localhost:8000/docs for API docs")
        print()
    
    def wait_for_shutdown(self):
        """Wait for shutdown signal."""
        try:
            print("✅ Bloomberg Terminal is now running!")
            self.print_status()
            
            # Start monitoring in background
            monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
            monitor_thread.start()
            
            # Wait for shutdown
            while self.is_running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            pass
    
    def shutdown(self):
        """Shutdown all processes."""
        print("\n🛑 Shutting down Bloomberg Terminal...")
        self.is_running = False
        
        for name, process in self.processes:
            try:
                print(f"   Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"   Force killing {name}...")
                    process.kill()
                    
                print(f"   ✅ {name} stopped")
                
            except Exception as e:
                print(f"   ⚠️  Error stopping {name}: {e}")
        
        print("✅ Bloomberg Terminal shutdown complete")
    
    def launch(self):
        """Main launch sequence."""
        try:
            self.print_banner()
            self.setup_signal_handlers()
            
            # Check dependencies
            if not self.check_dependencies():
                return False
            
            # Start services
            self.is_running = True
            
            # Start API server (optional)
            self.start_api_server()
            
            # Start terminal (required)
            if not self.start_terminal():
                self.is_running = False
                return False
            
            # Open browser
            self.open_browser()
            
            # Wait for shutdown
            self.wait_for_shutdown()
            
            return True
            
        except Exception as e:
            print(f"❌ Launch failed: {e}")
            self.shutdown()
            return False

def main():
    """Main entry point."""
    launcher = TerminalLauncher()
    
    try:
        success = launcher.launch()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        launcher.shutdown()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        launcher.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()