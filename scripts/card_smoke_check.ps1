param(
    [string]$BaseUrl = "http://127.0.0.1:8080",
    [string]$SessionId = "card-smoke-default",
    [ValidateSet("shadow", "on")]
    [string]$CardProcessingMode = "shadow",
    [string]$PromptText = "Write a Rust function named parse_port that trims whitespace, rejects empty input, rejects non-numeric input, rejects values outside 1..65535, does not panic, and includes tests. Output Rust code only.",
    [string]$PromptMode = "developer",
    [int]$MaxTokens = 256,
    [int]$PollSeconds = 180,
    [string]$OutputRoot = "C:\Users\micha\repos\airframe\artifacts\card_smoke"
)

$ErrorActionPreference = "Stop"

function Wait-ServerReady {
    param([string]$ReadyBaseUrl)

    $readyUrl = "$ReadyBaseUrl/api/repro/queue"
    for ($i = 0; $i -lt 120; $i++) {
        try {
            $null = Invoke-RestMethod -Method Get -Uri $readyUrl
            return
        } catch {
            Start-Sleep -Seconds 1
        }
    }

    throw "Server did not become ready at $ReadyBaseUrl"
}

Wait-ServerReady -ReadyBaseUrl $BaseUrl

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $OutputRoot ("{0}_{1}" -f $CardProcessingMode, $stamp)
$null = New-Item -ItemType Directory -Force -Path $runDir

$body = @{
    task = "card_smoke"
    prompt = $PromptText
    session_id = $SessionId
    prompt_mode = $PromptMode
    card_processing_mode = $CardProcessingMode
    max_tokens = $MaxTokens
    temperature = 0.0
    top_p = 1.0
    repetition_penalty = 1.0
    seed = 7777
    stream = $false
} | ConvertTo-Json -Depth 8

Set-Content -Path (Join-Path $runDir "request.json") -Value $body -NoNewline

$submit = Invoke-RestMethod -Method Post -Uri "$BaseUrl/" -ContentType "application/json" -Body $body
if (-not $submit.job_id) {
    throw "Server did not return a job_id. Response: $($submit | ConvertTo-Json -Compress)"
}

$jobId = [string]$submit.job_id
$statusUrl = "$BaseUrl/api/repro/job-status?job_id=$jobId"
$sessionCardUrl = "$BaseUrl/api/repro/session-card?session_id=$SessionId"
$pollIntervalSeconds = 5
$maxPolls = [Math]::Ceiling($PollSeconds / $pollIntervalSeconds)

$status = $null
for ($i = 0; $i -lt $maxPolls; $i++) {
    $status = Invoke-RestMethod -Method Get -Uri $statusUrl
    if ($status.status -eq "completed" -or $status.status -eq "failed") {
        break
    }
    Start-Sleep -Seconds $pollIntervalSeconds
}

if ($null -eq $status) {
    throw "No status received for job $jobId"
}

$status | ConvertTo-Json -Depth 10 | Set-Content -Path (Join-Path $runDir "status.json")

if ($status.status -ne "completed") {
    throw "Job $jobId did not complete successfully. Final status: $($status.status) Error: $($status.error)"
}

$sessionCard = Invoke-RestMethod -Method Get -Uri $sessionCardUrl
$sessionCard | ConvertTo-Json -Depth 10 | Set-Content -Path (Join-Path $runDir "session_card.json")

if ($status.result -and $status.result.text) {
    Set-Content -Path (Join-Path $runDir "response.txt") -Value ([string]$status.result.text) -NoNewline
}

[pscustomobject]@{
    mode = $CardProcessingMode
    session_id = $SessionId
    job_id = $jobId
    status = [string]$status.status
    stop_reason = [string]$status.result.stop_reason
    tokens_generated = [int]$status.result.tokens_generated
    card_present = ($null -ne $sessionCard.active_card)
    card_history_count = @($sessionCard.card_history).Count
    mirror_file = "artifacts/session_cards/$SessionId.json"
    run_dir = $runDir
} | ConvertTo-Json -Compress