@echo off
setlocal

rem Quick launcher for the SynthStream desktop GUI.
set "PROJECT_ROOT=%~dp0"
set "PYTHON_EXE="
set "PYTHON_ARGS="

rem Prefer a project virtual environment, then the Codex bundled runtime.
if exist "%PROJECT_ROOT%.venv\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" set "PYTHON_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

rem Fall back to a normal Windows Python installation.
if not defined PYTHON_EXE (
    where py >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3"
    ) else (
        set "PYTHON_EXE=python"
    )
)

set "PYTHONPATH=%PROJECT_ROOT%src;%PYTHONPATH%"
%PYTHON_EXE% %PYTHON_ARGS% -m synthstream.app %*
if errorlevel 1 (
    echo.
    echo SynthStream could not start. Check that Python and project dependencies are installed.
    pause
)

endlocal
