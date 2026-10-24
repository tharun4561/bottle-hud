@echo off
rem Starts the demo: full screen, live webcam. Double-click, or run it
rem from any terminal; extra options are passed on (e.g. demo.bat --screen 2).
cd /d "%~dp0"
rem Never look for the internet at the venue (Ultralytics checks at startup).
set YOLO_OFFLINE=1
echo Starting the demo (about 15 seconds)...
".venv\Scripts\python.exe" main.py --demo %*
rem Keep this window open after a problem, so the message can be read.
if errorlevel 1 pause
