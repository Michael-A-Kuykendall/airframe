# test_model.ps1 - Test a single GGUF model end-to-end
# Usage: .\scripts\test_model.ps1 -ModelPath "D:\shimmy-test-models\..."
# Exit: 0=PASS, 1=FAIL/CRASH, 2=OOM

param([Parameter(Mandatory=$true)][string]$ModelPath)

$SHIMMY     = "C:\Users\micha\repos\shimmy\target\debug\shimmy.exe"
$MAX_TOKENS = 20

# 1. File checks
if (-not (Test-Path $ModelPath)) { Write-Host "SKIP - not found"; exit 0 }
$sizeMB = [int]((Get-Item $ModelPath).Length / 1MB)
if ($sizeMB -gt 2000) { Write-Host "OOM  - ${sizeMB}MB > 2GB cap"; exit 2 }
$label = [System.IO.Path]::GetFileNameWithoutExtension($ModelPath)
Write-Host "Testing: $label (${sizeMB} MB)"

# 2. Kill all shimmy processes cleanly before starting
Get-Process shimmy -ErrorAction SilentlyContinue | ForEach-Object {
    $_.Kill()
    $_.WaitForExit(3000)
}
Start-Sleep -Seconds 3

# 3. Pick a free port
$sock = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
$sock.Start(); $PORT = $sock.LocalEndpoint.Port; $sock.Stop()
$URL = "http://127.0.0.1:$PORT"
Write-Host "  Port: $PORT"

# 4. Start shimmy
$env:SHIMMY_MAX_CTX = "2048"
$proc = Start-Process -FilePath $SHIMMY `
    -ArgumentList "serve","--bind","127.0.0.1:$PORT","--model-path",$ModelPath `
    -PassThru -NoNewWindow `
    -RedirectStandardOutput "$env:TEMP\shimmy_out.txt" `
    -RedirectStandardError  "$env:TEMP\shimmy_err.txt"
Write-Host "  PID: $($proc.Id)"

# Give process 2s to actually bind the port before polling
Start-Sleep -Seconds 2

# 5. Poll /health (blocking, up to 90s)
$ready = $false
for ($i = 2; $i -lt 90; $i++) {
    if ($proc.HasExited) {
        $err = (Get-Content "$env:TEMP\shimmy_err.txt" -ErrorAction SilentlyContinue) -join " "
        Write-Host "CRASH - died at ${i}s. $($err[-200..-1] -join '')"
        exit 1
    }
    try {
        $r = Invoke-WebRequest -Uri "$URL/health" -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true; Write-Host "  Ready after ${i}s"; break }
    } catch { }
    Start-Sleep -Seconds 1
}
if (-not $ready) {
    Write-Host "TIMEOUT - not ready after 90s"
    $proc | Stop-Process -Force -ErrorAction SilentlyContinue
    exit 1
}

# 6. Get model name — use the filename stem, avoid fallback to unrelated discovered models
$expectedModel = $label
try {
    $allModels = (Invoke-RestMethod -Uri "$URL/v1/models" -TimeoutSec 5).data | ForEach-Object { $_.id }
    Write-Host "  Available: $($allModels -join ', ')"
    # Best match: exact > partial filename > first non-phi non-default model
    $m = ($allModels | Where-Object { $_ -eq $expectedModel }) | Select-Object -First 1
    if (-not $m) { $m = ($allModels | Where-Object { $_ -like "*$expectedModel*" }) | Select-Object -First 1 }
    if (-not $m) { $m = ($allModels | Where-Object { $_ -notlike "*phi*" -and $_ -notlike "*default*" }) | Select-Object -First 1 }
    if (-not $m) { $m = $allModels[0] }
    Write-Host "  Using: $m"
} catch {
    Write-Host "FAIL - /v1/models: $_"
    $proc | Stop-Process -Force -ErrorAction SilentlyContinue; exit 1
}

# 7. Inference
$body = (@{ model=$m; messages=@(@{role="user";content="hi"}); stream=$false; max_tokens=$MAX_TOKENS } | ConvertTo-Json)
try {
    $resp    = Invoke-RestMethod -Uri "$URL/v1/chat/completions" -Method POST -ContentType "application/json" -Body $body -TimeoutSec 120
    $content = $resp.choices[0].message.content
    $finish  = $resp.choices[0].finish_reason
    $proc | Stop-Process -Force -ErrorAction SilentlyContinue
    if ([string]::IsNullOrWhiteSpace($content) -or $content.Length -lt 2) {
        Write-Host "WEAK - '$content' (finish=$finish)"; exit 1
    }
    Write-Host "PASS - '$content' (finish=$finish)"; exit 0
} catch {
    Write-Host "FAIL - inference: $_"
    $proc | Stop-Process -Force -ErrorAction SilentlyContinue; exit 1
}
