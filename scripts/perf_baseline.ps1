<#
.SYNOPSIS
    Tokens/sec benchmark against a running shimmy_server_gpu instance.

.DESCRIPTION
    Assumes server is already running on BaseUrl.
    Sends Runs requests per model, measures wall-clock time for each,
    computes tokens/sec from usage.completion_tokens.
    Outputs results to artifacts/perf_baseline_<timestamp>.json and .csv.

.EXAMPLE
    # Server already running on :8080
    powershell -ExecutionPolicy Bypass -File scripts\perf_baseline.ps1

    # Custom model list or repetitions
    powershell -ExecutionPolicy Bypass -File scripts\perf_baseline.ps1 -Runs 5
#>
param(
    [string]$BaseUrl     = "http://127.0.0.1:8080",
    [int]$Runs           = 3,
    [int]$MaxTokens      = 64,
    [string]$OutputDir   = "$PSScriptRoot\..\artifacts\perf_baseline",
    [string]$Label       = ""   # Optional label e.g. "before" or "after"
)

$ErrorActionPreference = "Stop"
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$null = New-Item -ItemType Directory -Force -Path $OutputDir

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$tag       = if ($Label) { "${Label}_${timestamp}" } else { $timestamp }
$jsonFile  = Join-Path $OutputDir "perf_${tag}.json"
$csvFile   = Join-Path $OutputDir "perf_${tag}.csv"

# Models to benchmark — same as smoke verified list
$Models = @(
    @{ Name = "TinyLlama-1.1B Q4_0";   Prompt = "Tell me a short story about a robot." },
    @{ Name = "Llama-3.2-1B Q4_K_M";   Prompt = "Tell me a short story about a robot." },
    @{ Name = "Llama-3.2-3B Q4_K_M";   Prompt = "Tell me a short story about a robot." }
)

# --- Readiness check ---
Write-Host "Checking server readiness at $BaseUrl ..."
$ready = $false
for ($i = 0; $i -lt 10; $i++) {
    try {
        $null = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/repro/queue" -TimeoutSec 3
        $ready = $true; break
    } catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) {
    Write-Error "Server not ready at $BaseUrl. Start the server first."
    exit 1
}
Write-Host "Server ready." -ForegroundColor Green

# --- Benchmark ---
$allResults = @()

foreach ($model in $Models) {
    Write-Host ""
    Write-Host "=== $($model.Name) ===" -ForegroundColor Cyan
    $runResults = @()

    for ($r = 1; $r -le $Runs; $r++) {
        $body = @{
            model       = "local"
            messages    = @(@{ role = "user"; content = $model.Prompt })
            max_tokens  = $MaxTokens
            temperature = 0.0
            stream      = $false
        } | ConvertTo-Json -Depth 5

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $resp = Invoke-RestMethod `
                -Method Post `
                -Uri "$BaseUrl/v1/chat/completions" `
                -ContentType "application/json" `
                -Body $body `
                -TimeoutSec 120
            $sw.Stop()

            $completionTokens = $resp.usage.completion_tokens
            $elapsedMs        = $sw.ElapsedMilliseconds
            $tokPerSec        = if ($elapsedMs -gt 0) { [math]::Round($completionTokens / ($elapsedMs / 1000.0), 2) } else { 0 }

            Write-Host ("  Run {0}: {1} tokens in {2}ms = {3} tok/s" -f $r, $completionTokens, $elapsedMs, $tokPerSec)
            $runResults += @{
                run             = $r
                completion_tokens = $completionTokens
                elapsed_ms      = $elapsedMs
                tokens_per_sec  = $tokPerSec
            }
        } catch {
            $sw.Stop()
            Write-Host ("  Run {0}: FAILED — {1}" -f $r, $_.Exception.Message) -ForegroundColor Red
        }
    }

    # Aggregate
    if ($runResults.Count -gt 0) {
        $avgTps  = [math]::Round(($runResults | Measure-Object -Property tokens_per_sec -Average).Average, 2)
        $maxTps  = [math]::Round(($runResults | Measure-Object -Property tokens_per_sec -Maximum).Maximum, 2)
        $minTps  = [math]::Round(($runResults | Measure-Object -Property tokens_per_sec -Minimum).Minimum, 2)
        Write-Host ("  -> avg={0}  min={1}  max={2} tok/s" -f $avgTps, $minTps, $maxTps) -ForegroundColor Yellow
    } else {
        $avgTps = $maxTps = $minTps = 0
    }

    $allResults += @{
        model        = $model.Name
        runs         = $runResults
        avg_tps      = $avgTps
        min_tps      = $minTps
        max_tps      = $maxTps
        label        = $tag
        max_tokens   = $MaxTokens
        timestamp    = $timestamp
    }
}

# --- Write outputs ---
$allResults | ConvertTo-Json -Depth 8 | Set-Content -Path $jsonFile -Encoding UTF8
Write-Host ""
Write-Host "Results written to: $jsonFile" -ForegroundColor Green

# CSV summary
$csvLines = @("label,model,avg_tps,min_tps,max_tps,max_tokens,timestamp")
foreach ($r in $allResults) {
    $csvLines += "$tag,$($r.model),$($r.avg_tps),$($r.min_tps),$($r.max_tps),$($r.max_tokens),$timestamp"
}
$csvLines | Set-Content -Path $csvFile -Encoding UTF8
Write-Host "CSV written to:     $csvFile" -ForegroundColor Green

# --- Summary table ---
Write-Host ""
Write-Host "SUMMARY ($tag)" -ForegroundColor Cyan
Write-Host ("{0,-28} {1,10} {2,10} {3,10}" -f "Model", "Avg tok/s", "Min", "Max")
Write-Host ("-" * 62)
foreach ($r in $allResults) {
    Write-Host ("{0,-28} {1,10} {2,10} {3,10}" -f $r.model, $r.avg_tps, $r.min_tps, $r.max_tps)
}
