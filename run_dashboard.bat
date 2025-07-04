@echo off
echo Starting Semantic Flow Analyzer Dashboard...
cd /d %~dp0
streamlit run examples/dashboard_launcher.py
pause