param(
    [string]$BaseUrl = "http://127.0.0.1:8080",
    [string]$RequestPath = "C:\Users\micha\repos\airframe\artifacts\story_4k_exact_request_nostream.json",
    [string]$OutputFile = "C:\Users\micha\repos\airframe\artifacts\story_4k_exact_current_status_new_text.txt",
    [int]$PollSeconds = 300
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
Write-Host "Job ID: $jobId"

$statusUrl = "$BaseUrl/api/repro/job-status?job_id=$jobId"

$status = $null
$lastObservedStatus = $null
$consecutivePollFailures = 0
$maxConsecutivePollFailures = 3
$pollIntervalSeconds = 5
$maxPolls = [Math]::Ceiling($PollSeconds / $pollIntervalSeconds)
for ($i = 0; $i -lt $maxPolls; $i++) {
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
    Start-Sleep -Seconds $pollIntervalSeconds
}

if ($null -eq $status) {
    throw "No status received for job $jobId"
}

if ($status.status -ne "completed") {
    if ($status.status -eq "running" -or $status.status -eq "queued") {
        throw "Job $jobId did not finish within ${PollSeconds}s. Last status: $($status.status)"
    }
    throw "Job $jobId did not complete successfully. Final status: $($status.status) Error: $($status.error)"
}

if ($status.result -and $status.result.text) {
    $text = [string]$status.result.text
    Set-Content -Path $OutputFile -Value $text -NoNewline
    Write-Host "Success! Text written to $OutputFile"
    [pscustomobject]@{
        job_id = $jobId
        status = [string]$status.status
        stop_reason = [string]$status.result.stop_reason
        tokens_generated = [int]$status.result.tokens_generated
        chars = $text.Length
        output_file = $OutputFile
    } | ConvertTo-Json -Compress
} else {
    throw "Completed job $jobId returned no text."
}
