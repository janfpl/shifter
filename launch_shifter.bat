@echo off
setlocal

rem ================================================================
rem  USER SETTINGS - the only lines you should need to edit
rem ================================================================

rem  Name of the conda environment shifter is installed in
rem  (or the full path to it, e.g. D:\envs\shifter).
set "ENV_NAME=shifter"

rem  Folder where Anaconda / Miniconda / Miniforge is installed, e.g.
rem      C:\Users\yourname\anaconda3
rem      C:\ProgramData\anaconda3
rem  Leave empty to auto-detect (checks PATH, then common locations).
rem  To find it: open Anaconda Prompt and run   where conda
set "CONDA_ROOT="

rem ================================================================
rem  Nothing below here should need changing
rem ================================================================

if defined CONDA_ROOT goto :check_root

rem --- 1. conda already on PATH? ---
for /f "delims=" %%F in ('where conda 2^>nul') do (
    if not defined CONDA_ROOT call :root_from_exe "%%F"
)
if defined CONDA_ROOT goto :check_root

rem --- 2. common install locations ---
for %%D in (
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\miniforge3"
    "%LOCALAPPDATA%\anaconda3"
    "%LOCALAPPDATA%\miniconda3"
    "C:\ProgramData\anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\miniforge3"
) do (
    if not defined CONDA_ROOT if exist "%%~D\Scripts\activate.bat" set "CONDA_ROOT=%%~D"
)
if defined CONDA_ROOT goto :check_root

echo ERROR: Could not find an Anaconda / Miniconda installation.
echo Open this file in Notepad and set CONDA_ROOT at the top.
goto :fail

:check_root
if exist "%CONDA_ROOT%\Scripts\activate.bat" goto :activate
echo ERROR: No conda installation found at:
echo     %CONDA_ROOT%
echo Check the CONDA_ROOT setting at the top of this file.
goto :fail

:activate
echo Using conda at: %CONDA_ROOT%
echo Activating environment: %ENV_NAME%
call "%CONDA_ROOT%\Scripts\activate.bat" "%ENV_NAME%"
if errorlevel 1 goto :bad_env

python -c "import shifter" >nul 2>&1
if errorlevel 1 goto :no_shifter

echo Starting shifter...
python -m shifter
if errorlevel 1 goto :fail
exit /b 0

:bad_env
echo ERROR: Could not activate the conda environment "%ENV_NAME%".
echo Check the ENV_NAME setting at the top of this file.
echo To list your environments: open Anaconda Prompt and run   conda env list
goto :fail

:no_shifter
echo ERROR: shifter is not installed in the environment "%ENV_NAME%".
echo See the Installation section of README.md.
goto :fail

:fail
echo.
pause
exit /b 1

rem ----------------------------------------------------------------
rem  Given the path to conda.exe / conda.bat, find the install root
rem  (the folder containing Scripts\activate.bat). conda lives in
rem  <root>\condabin, <root>\Scripts or <root>\Library\bin.
rem ----------------------------------------------------------------
:root_from_exe
set "_dir=%~dp1"
for %%P in ("%_dir%..") do set "_root=%%~fP"
if exist "%_root%\Scripts\activate.bat" (set "CONDA_ROOT=%_root%" & goto :eof)
for %%P in ("%_dir%..\..") do set "_root=%%~fP"
if exist "%_root%\Scripts\activate.bat" set "CONDA_ROOT=%_root%"
goto :eof
