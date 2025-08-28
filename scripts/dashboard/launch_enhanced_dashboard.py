#!/usr/bin/env python3
"""
Launch script for the Enhanced AI-Powered Underwriting Dashboard

This script launches the comprehensive dashboard showcasing:
- Expanded free review dataset capabilities
- 66 enhanced features vs 38 baseline
- Real-time risk assessment
- Model performance analytics
- Portfolio management tools
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'plotly', 
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def launch_dashboard():
    """Launch the enhanced dashboard."""
    print("🚀 Launching Enhanced AI-Powered Underwriting Dashboard")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Get project root and dashboard path
    project_root = Path(__file__).parent.parent.parent
    dashboard_path = project_root / 'src/dashboard/enhanced_app.py'
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return
    
    print("✅ Requirements check passed")
    print(f"📂 Dashboard location: {dashboard_path}")
    print("🌐 Starting Streamlit server...")
    print("\n" + "=" * 60)
    print("🎯 Dashboard Features:")
    print("   • Executive Overview - Key metrics and insights")
    print("   • Business Risk Assessment - Individual business analysis") 
    print("   • Model Performance Analytics - Baseline vs Enhanced comparison")
    print("   • Enhanced Features Showcase - 28 new sentiment features")
    print("   • Portfolio Management - Risk-return analysis")
    print("   • Data Pipeline Status - System monitoring")
    print("=" * 60)
    print("\n🚀 Opening dashboard in your browser...")
    print("⏹️  Press Ctrl+C to stop the dashboard\n")
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            str(dashboard_path),
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {str(e)}")

def main():
    """Main execution function."""
    print("🏦 AI-Powered Underwriting Engine")
    print("Enhanced Dashboard Launcher")
    print("-" * 40)
    
    launch_dashboard()

if __name__ == "__main__":
    main()