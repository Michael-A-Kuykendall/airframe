param(
    [string]$ServerBin   = "$PSScriptRoot\..\target\release\shimmy_server_gpu.exe",
    [string]$ModelDir    = "D:\shimmy-test-models\gguf_collection",
    [string]$BaseUrl     = "http://127.0.0.1:8080",
    [int]$StartupTimeout = 180,
    [int]$RequestTimeout = 180,
    [string]$OutputDir   = "$PSScriptRoot\..\artifacts\model_smoke",
    [switch]$IncludeLarge  # Pass -IncludeLarge to also test 7B models
)

$ErrorActionPreference = "Stop"

# Normalize paths so Start-Process redirect files resolve without '..' components
# (Win32 CreateProcess does not resolve '..' in redirect paths)
$ServerBin = [System.IO.Path]::GetFullPath($ServerBin)
$OutputDir  = [System.IO.Path]::GetFullPath($OutputDir)

# Verified models (quant_verify confirmed on RTX 3060).
# Each entry: @(filename, expected_keyword_in_response, prompt)
$VerifiedModels = @(
    @("TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf",     "Paris",  "The capital of France is"),
    @("Llama-3.2-1B-Instruct-Q4_K_M.gguf",      "Paris",  "The capital of France is"),
    @("Llama-3.2-3B-Instruct-Q4_K_M.gguf",      "Paris",  "The capital of France is"),
    @("phi-2.Q4_K_M.gguf",                       "Paris",  "The capital of France is"),
    @("gemma-2-2b-it-Q4_K_M.gguf",              "Paris",  "The capital of France is"),
    @("starcoder2-3b-Q4_K_M.gguf",              "def ",   "def hello_world():"),
    @("gpt2.Q4_K_M.gguf",                        "",       "The capital of France is")
)

# 7B models — larger VRAM needed, run only with -IncludeLarge
$LargeModels = @(
    @("deepseek-llm-7b-chat.Q4_K_M.gguf",               "Paris",  "The capital of France is"),
    @("deepseek-coder-6.7b-instruct.Q4_K_M.gguf",       "def ",   "def hello_world():"),
    @("qwen2-7b-instruct-q4_k_m.gguf",                  "Paris",  "The capital of France is")
)

$Models = if ($IncludeLarge) { $VerifiedModels + $LargeModels } else { $VerifiedModels }

$Prompt = "The capital of France is"
$CodePrompt = "def hello_world():"

$null = New-Item -ItemType Directory -Force -Path $OutputDir
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile   = Join-Path $OutputDir "smoke_$timestamp.log"
$results   = @()

function Write-Log {
    param([string]$Msg)
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $Msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

function Wait-ServerReady {
    param([string]$Url, [int]$Timeout)
    $readyUrl = "$Url/api/repro/queue"
    for ($i = 0; $i -lt $Timeout; $i++) {
        try {
            $null = Invoke-RestMethod -Method Get -Uri $readyUrl -TimeoutSec 2
            return $true
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    return $false
}

function Stop-ServerProcess {
    param($Proc)
    if ($Proc -and -not $Proc.HasExited) {
        $Proc.Kill()
        $Proc.WaitForExit(5000) | Out-Null
    }
}

Write-Log "=== Airframe model smoke test ==="
Write-Log "Models dir : $ModelDir"
Write-Log "Server bin : $ServerBin"
Write-Log "Base URL   : $BaseUrl"
Write-Log ""

foreach ($entry in $Models) {
    $modelFile    = $entry[0]
    $expectWord   = $entry[1]
    $promptText   = $entry[2]
    $modelPath    = Join-Path $ModelDir $modelFile

    if (-not (Test-Path $modelPath)) {
        Write-Log "SKIP  $modelFile (not found at $modelPath)"
        $results += [pscustomobject]@{ Model=$modelFile; Result="SKIP"; Detail="file not found" }
        continue
    }

    Write-Log "START $modelFile"
    # No output redirection — server stdout/stderr flows to this terminal so failures are visible.
    $procArgs = @{
        FilePath    = $ServerBin
        PassThru    = $true
        NoNewWindow = $true
    }

    $env:LIBSHIMMY_MODEL_PATH = $modelPath
    $env:SHIMMY_PORT          = "8080"
    $proc = Start-Process @procArgs

    $ready = Wait-ServerReady -Url $BaseUrl -Timeout $StartupTimeout
    if (-not $ready) {
        Write-Log "FAIL  $modelFile — server did not become ready in ${StartupTimeout}s"
        $results += [pscustomobject]@{ Model=$modelFile; Result="FAIL"; Detail="startup timeout" }
        Stop-ServerProcess $proc
        continue
    }

    # Each model entry carries its own prompt
    $body = @{
        model       = "local"
        messages    = @(@{ role = "user"; content = $promptText })
        max_tokens  = 32
        temperature = 0.0
        stream      = $false
    } | ConvertTo-Json -Depth 6

    $response = $null
    try {
        $response = Invoke-RestMethod `
            -Method Post `
            -Uri "$BaseUrl/v1/chat/completions" `
            -ContentType "application/json" `
            -Body $body `
            -TimeoutSec $RequestTimeout
    } catch {
        Write-Log "FAIL  $modelFile — request error: $_"
        $results += [pscustomobject]@{ Model=$modelFile; Result="FAIL"; Detail="request error: $_" }
        Stop-ServerProcess $proc
        continue
    }

    $text = ""
    try { $text = $response.choices[0].message.content } catch {}

    if ($text.Length -gt 0) {
        $pass = ($expectWord -eq "") -or ($text -match [regex]::Escape($expectWord))
        $tag  = if ($pass) { "PASS" } else { "WEAK" }
        Write-Log "$tag  $modelFile — response: $($text.Substring(0, [Math]::Min(80, $text.Length)))"
        $results += [pscustomobject]@{ Model=$modelFile; Result=$tag; Detail=$text }
    } else {
        Write-Log "FAIL  $modelFile — empty response"
        $results += [pscustomobject]@{ Model=$modelFile; Result="FAIL"; Detail="empty response" }
    }

    Stop-ServerProcess $proc
    Start-Sleep -Seconds 2
}

Write-Log ""
Write-Log "=== Summary ==="
$pass = ($results | Where-Object Result -eq "PASS").Count
$weak = ($results | Where-Object Result -eq "WEAK").Count
$fail = ($results | Where-Object Result -eq "FAIL").Count
$skip = ($results | Where-Object Result -eq "SKIP").Count
Write-Log "PASS: $pass  WEAK: $weak  FAIL: $fail  SKIP: $skip  Total: $($results.Count)"

$csvPath = Join-Path $OutputDir "smoke_$timestamp.csv"
$results | Export-Csv -Path $csvPath -NoTypeInformation
Write-Log "Results written to: $csvPath"

if ($fail -gt 0) {
    Write-Log "SMOKE TEST: FAILED ($fail failures)"
    exit 1
}
Write-Log "SMOKE TEST: PASSED"
exit 0
