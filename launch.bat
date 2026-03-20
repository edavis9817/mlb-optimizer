@echo off
cd /d "%~dp0"
echo Starting MLB Toolbox...
echo.
python -m streamlit run app/streamlit_app.py
pause
