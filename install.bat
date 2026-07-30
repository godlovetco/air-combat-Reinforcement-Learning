@echo off
REM One-click installer for the UCAV AI Pilot DCS addon (Windows).
REM Double-click this file, or run it from a command prompt.
setlocal
cd /d "%~dp0"

REM Prefer the Windows "py" launcher, fall back to "python".
where py >nul 2>nul && (set "PY=py") || (set "PY=python")

%PY% -m dcs_bridge.install %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo Install finished.
) else (
    echo Installer exited with error %RC%.
)
echo Press any key to close...
pause >nul
endlocal
