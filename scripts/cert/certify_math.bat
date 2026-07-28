@echo off
REM ============================================================================
REM  MATH CERT — plan vs peel + quant_verify → reds.json
REM  See CERT_REGIMEN.md
REM
REM  Usage:
REM    scripts\certify_math.bat <family-id> <path-to.gguf> ["multi-token prompt"] [--skip-gpu]
REM
REM  --skip-gpu: only judge an existing peel (PKG\peel.json or obs path)
REM ============================================================================
setlocal EnableExtensions EnableDelayedExpansion

if "%~1"=="" goto :usage
if "%~2"=="" goto :usage

set "FAMILY_ID=%~1"
set "GGUF=%~2"
set "PROMPT=%~3"
if "%PROMPT%"=="" set "PROMPT=The capital of France is"
set "SKIP_GPU=0"
if /I "%~3"=="--skip-gpu" (
  set "PROMPT=The capital of France is"
  set "SKIP_GPU=1"
)
if /I "%~4"=="--skip-gpu" set "SKIP_GPU=1"

if not defined SHIMMY_MAX_CTX set "SHIMMY_MAX_CTX=8192"

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.."
set "WS=%CD%"
popd

set "AF=%WS%\airframe"
set "PKG=%WS%\cert\packages\%FAMILY_ID%"
set "MATH=%PKG%\math"
set "PY=python"

mkdir "%PKG%" 2>nul
mkdir "%MATH%" 2>nul

echo === certify_math %FAMILY_ID% ===
echo GGUF=%GGUF%
echo PROMPT=%PROMPT%
echo MATH=%MATH%

REM Unit regression first (locks the judge)
echo [T0] cert_reds_test.py
"%PY%" "%WS%\scripts\cert_reds_test.py"
if errorlevel 1 (
  echo RED T0 unit tests
  exit /b 1
)

if "%SKIP_GPU%"=="1" goto :judge

if not exist "%GGUF%" (
  echo ERROR: GGUF not found: %GGUF%
  exit /b 1
)

echo [G0] ensure stack_dump_gpu + quant_verify built
cd /d "%AF%"
cargo build --features isf --release --bin stack_dump_gpu --bin quant_verify > "%MATH%\build.log" 2>&1
if errorlevel 1 (
  echo RED G0 build — see math\build.log
  exit /b 1
)

echo [G1] quant_verify
"%AF%\target\release\quant_verify.exe" -- --model-path "%GGUF%" > "%MATH%\quant_verify.log" 2>&1
REM do not exit on fail — reds will capture

echo [G2] stack_dump_gpu peel
"%AF%\target\release\stack_dump_gpu.exe" "%GGUF%" "%PROMPT%" "%MATH%\peel.json" > "%MATH%\peel.log" 2>&1
if errorlevel 1 (
  echo RED G2 peel failed — see math\peel.log
  exit /b 1
)

:judge
if not exist "%MATH%\peel.json" (
  if exist "%PKG%\obs\obs1\airframe.stack.peel.json" (
    copy /Y "%PKG%\obs\obs1\airframe.stack.peel.json" "%MATH%\peel.json" >nul
  ) else (
    echo ERROR: no peel.json — run without --skip-gpu first
    exit /b 1
  )
)

echo [G3] plan + reds
set "QLOG="
if exist "%MATH%\quant_verify.log" set "QLOG=--quant-log %MATH%\quant_verify.log"
"%PY%" "%WS%\scripts\cert_reds.py" "%MATH%\peel.json" --family-id "%FAMILY_ID%" --plan-out "%MATH%\plan.json" --reds-out "%MATH%\reds.json" --report-out "%MATH%\REPORT.md" %QLOG%
set "RC=%ERRORLEVEL%"

echo [G4] ledger
"%PY%" "%WS%\scripts\cert_ledger.py" record --family-id "%FAMILY_ID%" --reds-json "%MATH%\reds.json" --report "%MATH%\REPORT.md" --chat-ok unknown

echo === done RC=%RC% report=%MATH%\REPORT.md ===
exit /b %RC%

:usage
echo Usage: certify_math.bat ^<family-id^> ^<gguf^> ["prompt"] [--skip-gpu]
exit /b 2
