@echo off
title NagaWorks STEP File Editor V1.1
cd /d "%~dp0"
set QT_API=pyside6
python -m pip install -q -r requirements.txt
python step_file_editor.py
if errorlevel 1 pause
