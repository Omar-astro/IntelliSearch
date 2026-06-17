@echo off
title Project Launcher

set "PROJECT_ROOT=%~dp0.."

echo Starting Backend...
start "Backend Server" /D "%PROJECT_ROOT%" cmd /c "py backend\main.py"

echo Starting Frontend...
start "Angular Frontend" /D "%PROJECT_ROOT%\Frontend" cmd /c "npm.cmd start"

timeout /t 15 /nobreak > nul

start http://localhost:4200/

echo.
echo =====================================
echo Servers started!
echo Press any key to stop everything.
echo =====================================
pause > nul

echo Stopping servers...

taskkill /FI "WINDOWTITLE eq Backend Server" /T /F
taskkill /FI "WINDOWTITLE eq Angular Frontend" /T /F

echo Done.
exit
