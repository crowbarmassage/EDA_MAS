#!/usr/bin/env python3
"""
Streamlit App Launcher for Multi-Agent EDA System
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app."""
    print("🚀 Launching Multi-Agent EDA System - Streamlit App")
    print("=" * 60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version {streamlit.__version__} found")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully")
    
    # Check if plotly is installed
    try:
        import plotly
        print(f"✅ Plotly version {plotly.__version__} found")
    except ImportError:
        print("❌ Plotly not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
        print("✅ Plotly installed successfully")
    
    print("\n🌐 Starting Streamlit web application...")
    print("📱 The app will open in your default web browser")
    print("🔗 Local URL: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the application")
    print("=" * 60)
    
    # Launch Streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit app: {e}")
        print("💡 Try running manually: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
