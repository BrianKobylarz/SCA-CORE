#!/usr/bin/env python3
"""
Alternative dashboard launcher that properly invokes Streamlit
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit dashboard"""
    # Get the path to the dashboard script
    dashboard_path = Path(__file__).parent / "examples" / "dashboard_launcher.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard script not found at {dashboard_path}")
        sys.exit(1)
    
    # Change to the project directory
    os.chdir(Path(__file__).parent)
    
    print("🚀 Launching Semantic Flow Analyzer Dashboard...")
    print("=" * 60)
    print("The dashboard will open in your default web browser.")
    print("Press Ctrl+C in this terminal to stop the dashboard.")
    print("=" * 60)
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("\nMake sure Streamlit is installed:")
        print("  pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()