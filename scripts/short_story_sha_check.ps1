param(
    [string]$BaseUrl = "http://127.0.0.1:8080",
    [string]$RequestPath = "C:\Users\micha\repos\airframe\artifacts\story_seed7777_128tok_request.json",
    [string]$ExpectedSha = "f82a1ad07e5f74415a3121821e580998eecda4edd30b43efc9b294aa591c7974",
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"

function Invoke-JsonGet {
    param([string]$Uri)

    Invoke-RestMethod -Method Get -Uri $Uri -ErrorAction Stop
}

function Invoke-JsonPost {
    param(
        [string]$Uri,
        [string]$Body
    )

    Invoke-RestMethod -Method Post -Uri $Uri -ContentType "application/json" -Body $Body -ErrorAction Stop
}

$readyUrl = "$BaseUrl/api/repro/queue"
$ready = $false
for ($i = 0; $i -lt 120; $i++) {
    try {
        $null = Invoke-JsonGet -Uri $readyUrl
        $ready = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $ready) {
    throw "Server did not become ready at $BaseUrl"
}

$requestBody = Get-Content $RequestPath -Raw
$submit = Invoke-JsonPost -Uri "$BaseUrl/" -Body $requestBody

if (-not $submit.job_id) {
    throw "Server did not return a job_id. Response: $($submit | ConvertTo-Json -Compress)"
}

$jobId = [string]$submit.job_id
$statusUrl = "$BaseUrl/api/repro/job-status?job_id=$jobId"

$status = $null
$lastObservedStatus = $null
$consecutivePollFailures = 0
$maxConsecutivePollFailures = 3
for ($i = 0; $i -lt $PollSeconds; $i++) {
    try {
        $status = Invoke-JsonGet -Uri $statusUrl
        $consecutivePollFailures = 0
        $lastObservedStatus = $status.status
        if ($status.status -eq "completed" -or $status.status -eq "failed") {
            break
        }
    } catch {
        $consecutivePollFailures += 1
        if ($consecutivePollFailures -ge $maxConsecutivePollFailures) {
            $lastStatusText = if ($lastObservedStatus) { $lastObservedStatus } else { "none" }
            throw "Lost contact with server while polling job $jobId. Last observed status: $lastStatusText"
        }
    }
    Start-Sleep -Seconds 1
}

if ($null -eq $status) {
    throw "No status received for job $jobId"
}

if ($status.status -ne "completed") {
    throw "Job $jobId did not complete successfully. Final status: $($status.status)"
}

$text = ""
if ($status.result -and $status.result.text) {
    $text = [string]$status.result.text
}

$sha256 = [System.Security.Cryptography.SHA256]::Create()
$bytes = [System.Text.Encoding]::UTF8.GetBytes($text)
$hash = [System.BitConverter]::ToString($sha256.ComputeHash($bytes)).Replace('-', '').ToLowerInvariant()

[pscustomobject]@{
    job_id = $jobId
    status = [string]$status.status
    stop_reason = if ($status.result) { [string]$status.result.stop_reason } else { $null }
    tokens_generated = if ($status.result) { [int]$status.result.tokens_generated } else { $null }
    chars = $text.Length
    sha256 = $hash
    expected_sha256 = $ExpectedSha
    matches_expected = ($hash -eq $ExpectedSha)
} | ConvertTo-Json -Compress