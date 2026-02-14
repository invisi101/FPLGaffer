@echo off
title FPL Predictor - Installer
color 0A

:: Bulletproof wrapper: even if :main crashes, we always pause
call :main
echo.
pause
exit /b

:: ===================================================================
:main
:: ===================================================================
echo.
echo  ======================================
echo    FPL Points Predictor - Installer
echo  ======================================
echo.

set "INSTALL_DIR=%USERPROFILE%\fpl"

:: -------------------------------------------------------------------
:: Step 1: Check for Python
:: -------------------------------------------------------------------
echo [1/5] Checking for Python...

where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=py -3"
    goto :found_python
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON=python"
    goto :found_python
)

echo.
echo  Python is not installed.
echo.
echo  Please install Python manually:
echo    1. Go to https://www.python.org/downloads/
echo    2. Click the big yellow "Download Python" button
echo    3. Run the downloaded file
echo    4. IMPORTANT: Tick "Add Python to PATH" at the bottom
echo    5. Click "Install Now"
echo    6. Then close this window and run install-windows.bat again
echo.
goto :eof

:found_python
%PYTHON% --version
echo  Python found.
echo.

:: -------------------------------------------------------------------
:: Step 2: Download the project
:: -------------------------------------------------------------------
echo [2/5] Downloading FPL Predictor...

set "ZIP_URL=https://github.com/invisi101/xtifpl-mac/archive/refs/heads/master.zip"
set "ZIP_FILE=%TEMP%\fpl-download.zip"
set "EXTRACT_DIR=%TEMP%\fpl-extract"

:: Clean up any previous failed attempts
if exist "%ZIP_FILE%" del "%ZIP_FILE%" >nul 2>&1
if exist "%EXTRACT_DIR%" rmdir /S /Q "%EXTRACT_DIR%" >nul 2>&1

:: Try curl first (built into Windows 10+), then PowerShell as fallback
where curl.exe >nul 2>&1
if not errorlevel 1 (
    echo  Downloading with curl...
    curl.exe -L -o "%ZIP_FILE%" "%ZIP_URL%"
) else (
    echo  Downloading with PowerShell...
    powershell -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%ZIP_URL%' -OutFile '%ZIP_FILE%'"
)

if not exist "%ZIP_FILE%" (
    echo.
    echo  ERROR: Download failed. Check your internet connection.
    goto :eof
)

echo  Download complete. Extracting...

:: Try tar first (built into Windows 10+), then PowerShell as fallback
where tar.exe >nul 2>&1
if not errorlevel 1 (
    mkdir "%EXTRACT_DIR%" >nul 2>&1
    tar.exe -xf "%ZIP_FILE%" -C "%EXTRACT_DIR%"
) else (
    powershell -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%EXTRACT_DIR%' -Force"
)

if not exist "%EXTRACT_DIR%\xtifpl-mac-master" (
    echo.
    echo  ERROR: Extraction failed.
    goto :eof
)

if exist "%INSTALL_DIR%" (
    echo  Updating existing installation...
    robocopy "%EXTRACT_DIR%\xtifpl-mac-master" "%INSTALL_DIR%" /E /NFL /NDL /NJH /NJS >nul
) else (
    move "%EXTRACT_DIR%\xtifpl-mac-master" "%INSTALL_DIR%" >nul
)

rmdir /S /Q "%EXTRACT_DIR%" >nul 2>&1
del "%ZIP_FILE%" >nul 2>&1

echo  Installed to %INSTALL_DIR%
echo.

:: -------------------------------------------------------------------
:: Step 3: Create virtual environment
:: -------------------------------------------------------------------
echo [3/5] Setting up Python environment...

cd /d "%INSTALL_DIR%"

if not exist ".venv\Scripts\python.exe" (
    echo  Creating virtual environment...
    %PYTHON% -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo  ERROR: Failed to create virtual environment.
        goto :eof
    )
)

echo  Virtual environment ready.
echo.

:: -------------------------------------------------------------------
:: Step 4: Install dependencies
:: -------------------------------------------------------------------
echo [4/5] Installing dependencies (this may take a few minutes)...

.venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo  Dependencies installed.
echo.

:: -------------------------------------------------------------------
:: Step 5: Create desktop shortcut and launcher
:: -------------------------------------------------------------------
echo [5/5] Creating launcher...

:: Create the run script
echo @echo off > "%INSTALL_DIR%\run.bat"
echo chcp 65001 ^>nul 2^>^&1 >> "%INSTALL_DIR%\run.bat"
echo set PYTHONIOENCODING=utf-8 >> "%INSTALL_DIR%\run.bat"
echo title FPL Points Predictor >> "%INSTALL_DIR%\run.bat"
echo cd /d "%INSTALL_DIR%" >> "%INSTALL_DIR%\run.bat"
echo echo. >> "%INSTALL_DIR%\run.bat"
echo echo  Starting FPL Predictor... >> "%INSTALL_DIR%\run.bat"
echo echo  The app will open in your browser shortly. >> "%INSTALL_DIR%\run.bat"
echo echo  To stop the app, close this window. >> "%INSTALL_DIR%\run.bat"
echo echo. >> "%INSTALL_DIR%\run.bat"
echo start "" http://127.0.0.1:9876 >> "%INSTALL_DIR%\run.bat"
echo .venv\Scripts\python.exe -m src.app >> "%INSTALL_DIR%\run.bat"
echo echo. >> "%INSTALL_DIR%\run.bat"
echo echo  The app has stopped. >> "%INSTALL_DIR%\run.bat"
echo pause >> "%INSTALL_DIR%\run.bat"

:: Create desktop shortcut
powershell -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\FPL Predictor.lnk'); $s.TargetPath = '%INSTALL_DIR%\run.bat'; $s.WorkingDirectory = '%INSTALL_DIR%'; $s.Description = 'FPL Points Predictor'; $s.Save()" >nul 2>&1

if exist "%USERPROFILE%\Desktop\FPL Predictor.lnk" (
    echo  Shortcut created on Desktop.
) else (
    echo  Could not create shortcut. Run the app from:
    echo    %INSTALL_DIR%\run.bat
)

echo.
echo  ======================================
echo    Installation complete!
echo  ======================================
echo.
echo  To run: double-click "FPL Predictor" on your Desktop
echo  First time? Click "Refresh Data" then "Train Models" in the app.
echo.
goto :eof
