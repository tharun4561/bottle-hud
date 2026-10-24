@echo off
rem Backup demo: replays demo_clip.mp4 in a loop instead of the webcam.
rem Extra options are passed on (e.g. demo_backup.bat --screen 2).
cd /d "%~dp0"
set YOLO_OFFLINE=1
echo Starting the backup demo (about 15 seconds)...
".venv\Scripts\python.exe" main.py --demo --video demo_clip.mp4 --loop %*
if errorlevel 1 pause
