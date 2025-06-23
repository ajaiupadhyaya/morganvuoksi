#!/usr/bin/env python3
"""
MorganVuoksi Elite Terminal - Quick Launch Script
One-click launcher for the Bloomberg-grade quantitative finance platform.
"""

import os
import sys
import subprocess
import platform

def main():
    """Launch the MorganVuoksi Elite Terminal."""
    
    print("""
    
🚀 MORGANVUOKSI ELITE TERMINAL
=================================

Bloomberg-Grade Quantitative Finance Platform
    
Starting elite terminal in 3 seconds...
    """)
    
    import time
    time.sleep(3)
    
    try:
        # Make enhance_terminal.py executable and run it
        if platform.system() != "Windows":
            os.chmod("enhance_terminal.py", 0o755)
        
        # Launch the main terminal
        subprocess.run([sys.executable, "enhance_terminal.py"], check=True)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Launch cancelled by user.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n\n❌ Error launching terminal: {e}")
        print("\n💡 Try running manually:")
        print("   python enhance_terminal.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("\n💡 Try running manually:")
        print("   python enhance_terminal.py")
        sys.exit(1)

if __name__ == "__main__":
    main()