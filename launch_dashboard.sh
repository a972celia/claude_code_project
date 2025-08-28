#!/bin/bash

# Enhanced AI-Powered Underwriting Dashboard Launcher
# Simple script to launch the comprehensive dashboard

echo "🏦 AI-Powered Underwriting Engine"
echo "Enhanced Dashboard Launcher"
echo "================================"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit plotly
fi

echo "🚀 Launching Enhanced Dashboard..."
echo ""
echo "🎯 Dashboard Features:"
echo "   • Executive Overview"
echo "   • Business Risk Assessment" 
echo "   • Model Performance Analytics"
echo "   • Enhanced Features Showcase"
echo "   • Portfolio Management"
echo "   • Data Pipeline Status"
echo ""
echo "🌐 Opening in browser..."
echo "⏹️  Press Ctrl+C to stop"
echo ""

# Launch the enhanced dashboard
streamlit run src/dashboard/enhanced_app.py --server.port 8501 --browser.gatherUsageStats false