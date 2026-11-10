@echo off
REM Activa el entorno del proyecto para que "python" tenga Streamlit instalado.
call ".venv\Scripts\activate.bat"
python -m streamlit run app.py
pause
