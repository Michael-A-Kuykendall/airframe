#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Soak test for SHIMMY_KV_QUANT=int4 mode.
    Sends N consecutive requests at a given ctx to verify stability.

.DESCRIPTION
    Starts the server with INT4 KV mode, sends a configurable number of
    chat-completion requests, checks each response is non-empty, and reports
    pass/fail statistics.

.PARAMETER ModelPath
    Path to the GGUF model file. Defaults to Llama-3.2-3B-Instruct-Q4_K_M.gguf.

.PARAMETER Requests
    Number of consecutive requests to send. Default 10.

.PARAMETER Ctx
    Max context length for the server. Default 2048.

.PARAMETER PrefillChunk
    SHIMMY_PREFILL_CHUNK value. Default 64.

.PARAMETER Port
    Port for the server. Default 8099.

.PARAMETER RequestTimeout
    Per-request timeout in seconds. Default 600.

.EXAMPLE
    pwsh -ExecutionPolicy Bypass -File scripts/int4_soak_test.ps1
    pwsh -ExecutionPolicy Bypass -File scripts/int4_soak_test.ps1 -Requests 5 -Ctx 512
#>
param(
    [string]$ModelPath    = "D:\shimmy-test-models\gguf_collection\Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    [string]$ServerBin    = "$PSScriptRoot\..\target\release\shimmy_server_gpu.exe",
    [int]$Requests        = 10,
    [int]$Ctx             = 2048,
    [int]$PrefillChunk    = 64,
    [int]$Port            = 8099,
    [int]$RequestTimeout  = 600,
    [string]$OutDir       = "$PSScriptRoot\..\artifacts\soak"
)

$ErrorActionPreference = "Stop"
$ServerBin = [System.IO.Path]::GetFullPath($ServerBin)
$OutDir    = [System.IO.Path]::GetFullPath($OutDir)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$BaseUrl   = "http://127.0.0.1:$Port"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile   = Join-Path $OutDir "soak_int4_ctx${Ctx}_${Timestamp}.json"

Write-Host ""
Write-Host "=== INT4 KV Soak Test ===" -ForegroundColor Cyan
Write-Host "  Model:    $ModelPath"
Write-Host "  Requests: $Requests"
Write-Host "  Ctx:      $Ctx"
Write-Host "  Port:     $Port"
Write-Host "  Log:      $LogFile"
Write-Host ""

# --- Start server ---
Write-Host "[SOAK] Starting server..." -ForegroundColor Yellow

$env:SHIMMY_KV_QUANT     = "int4"
$env:SHIMMY_PORT         = "$Port"
$env:SHIMMY_MAX_CTX      = "$Ctx"
$env:SHIMMY_PREFILL_CHUNK = "$PrefillChunk"
$env:LIBSHIMMY_MODEL_PATH = $ModelPath
$env:RUST_BACKTRACE       = "1"

$ServerLog = Join-Path $OutDir "server_${Timestamp}.log"
$ServerProc = Start-Process -FilePath $ServerBin -NoNewWindow -PassThru `
    -RedirectStandardOutput $ServerLog -RedirectStandardError $ServerLog

Write-Host "[SOAK] Server PID: $($ServerProc.Id)"

# Wait for server ready
$Ready    = $false
$Waited   = 0
$MaxWait  = 180
while ($Waited -lt $MaxWait) {
    Start-Sleep -Seconds 5
    $Waited += 5
    $Content = Get-Content $ServerLog -ErrorAction SilentlyContinue -Raw
    if ($Content -match "Async Listener") {
        $Ready = $true
        break
    }
}

if (-not $Ready) {
    Write-Host "[SOAK] FAIL: Server did not start within ${MaxWait}s" -ForegroundColor Red
    $ServerProc | Stop-Process -Force -ErrorAction SilentlyContinue
    exit 1
}
Write-Host "[SOAK] Server ready." -ForegroundColor Green

# Short prompts that vary per request to stress KV cache independence
$Prompts = @(
    "Briefly, what is the boiling point of water in Celsius?",
    "Name the three primary colors.",
    "What is 7 multiplied by 8?",
    "In one sentence, what is photosynthesis?",
    "What element has the chemical symbol Fe?",
    "Name the largest planet in our solar system.",
    "What language is spoken in Brazil?",
    "What is the square root of 144?",
    "In which year did the First World War begin?",
    "What is the speed of light in meters per second (approximate)?"
)

$Results = @()
$Passed  = 0
$Failed  = 0

for ($i = 0; $i -lt $Requests; $i++) {
    $Prompt = $Prompts[$i % $Prompts.Count]
    Write-Host "[SOAK] Request $($i+1)/$Requests: `"$Prompt`"" -ForegroundColor Cyan

    $Body = @{
        model    = "airframe"
        messages = @(@{ role = "user"; content = $Prompt })
        max_tokens = 16
        temperature = 0.0
    } | ConvertTo-Json -Depth 5

    $StartTime = Get-Date
    try {
        $Resp = Invoke-RestMethod `
            -Uri "$BaseUrl/v1/chat/completions" `
            -Method POST `
            -ContentType "application/json" `
            -Body $Body `
            -TimeoutSec $RequestTimeout

        $Elapsed = ((Get-Date) - $StartTime).TotalSeconds
        $Text = $Resp.choices[0].message.content

        if ([string]::IsNullOrWhiteSpace($Text)) {
            Write-Host "  FAIL (empty response) elapsed=${Elapsed}s" -ForegroundColor Red
            $Results += [PSCustomObject]@{ request=$i+1; prompt=$Prompt; status="FAIL_EMPTY"; elapsed=$Elapsed; response="" }
            $Failed++
        } else {
            Write-Host "  PASS elapsed=${Elapsed}s  reply=`"$($Text.Substring(0, [Math]::Min(60, $Text.Length)))...`"" -ForegroundColor Green
            $Results += [PSCustomObject]@{ request=$i+1; prompt=$Prompt; status="PASS"; elapsed=$Elapsed; response=$Text }
            $Passed++
        }
    } catch {
        $Elapsed = ((Get-Date) - $StartTime).TotalSeconds
        Write-Host "  FAIL (exception) elapsed=${Elapsed}s  err=$_" -ForegroundColor Red
        $Results += [PSCustomObject]@{ request=$i+1; prompt=$Prompt; status="FAIL_EXCEPTION"; elapsed=$Elapsed; response="$_" }
        $Failed++
    }
}

# --- Stop server ---
$ServerProc | Stop-Process -Force -ErrorAction SilentlyContinue

# --- Write JSON ---
$Summary = [PSCustomObject]@{
    timestamp    = $Timestamp
    model        = $ModelPath
    ctx          = $Ctx
    requests     = $Requests
    passed       = $Passed
    failed       = $Failed
    results      = $Results
}
$Summary | ConvertTo-Json -Depth 10 | Set-Content $LogFile

Write-Host ""
Write-Host "=== Soak Test Summary ===" -ForegroundColor Cyan
Write-Host "  PASS: $Passed / $Requests" -ForegroundColor $(if ($Failed -eq 0) { "Green" } else { "Yellow" })
Write-Host "  FAIL: $Failed / $Requests" -ForegroundColor $(if ($Failed -gt 0) { "Red" } else { "Green" })
Write-Host "  Log:  $LogFile"
Write-Host ""

if ($Failed -gt 0) { exit 1 } else { exit 0 }
