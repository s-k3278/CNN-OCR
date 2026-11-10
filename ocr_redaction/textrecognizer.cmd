@echo off
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 "%~dp0textrecognizer.py" %*
) else (
  python "%~dp0textrecognizer.py" %*
)
