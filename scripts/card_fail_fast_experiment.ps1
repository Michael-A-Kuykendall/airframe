param(
    [string]$BaseUrl = "http://127.0.0.1:8080",
    [string]$PromptFile,
    [string]$PromptText,
    [string[]]$Modes = @("off", "instruction", "prefix", "bootstrap"),
    [string]$TaskPromptMode = "developer",
    [string]$CardPromptMode = "raw",
    [int]$PollSeconds = 300,
    [int]$TaskMaxTokens = 512,
    [int]$CardMaxTokens = 320,
    [switch]$NoCardLint,
    [string]$OutputRoot = "C:\Users\micha\repos\airframe\artifacts\card_fail_fast"
)

$ErrorActionPreference = "Stop"

function Get-TaskPrompt {
    if ($PromptText) {
        return $PromptText
    }

    if ($PromptFile) {
        if (-not (Test-Path $PromptFile)) {
            throw "Prompt file not found: $PromptFile"
        }
        return Get-Content $PromptFile -Raw
    }

    throw "Provide either -PromptText or -PromptFile"
}

function Wait-ServerReady {
    param([string]$BaseUrl)

    $readyUrl = "$BaseUrl/api/repro/queue"
    for ($i = 0; $i -lt 120; $i++) {
        try {
            $null = Invoke-RestMethod -Method Get -Uri $readyUrl
            return
        } catch {
            Start-Sleep -Seconds 1
        }
    }

    throw "Server did not become ready at $BaseUrl"
}

function Submit-GenerationJob {
    param(
        [string]$BaseUrl,
        [hashtable]$Body,
        [int]$PollSeconds
    )

    $json = $Body | ConvertTo-Json -Depth 8 -Compress
    $submit = Invoke-RestMethod -Method Post -Uri "$BaseUrl/" -ContentType "application/json" -Body $json
    if (-not $submit.job_id) {
        throw "Server did not return a job_id. Response: $($submit | ConvertTo-Json -Compress)"
    }

    $jobId = [string]$submit.job_id
    $statusUrl = "$BaseUrl/api/repro/job-status?job_id=$jobId"
    $status = $null
    $pollIntervalSeconds = 5
    $maxPolls = [Math]::Ceiling($PollSeconds / $pollIntervalSeconds)

    for ($i = 0; $i -lt $maxPolls; $i++) {
        try {
            $status = Invoke-RestMethod -Method Get -Uri $statusUrl
            if ($status.status -eq "completed" -or $status.status -eq "failed") {
                break
            }
        } catch {
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

    return $status
}

function New-CardPrompt {
    param([string]$TaskPrompt)

    @"
Compress the following task into a compact JSON task card.

Rules:
- Output JSON only.
- Do not output markdown fences.
- Keep fields compact and operational.
- Preserve hard constraints, hard prohibitions, exact inventories, assumptions, unresolved unknowns, the current step, and next steps.
- If a field has no values, use [] or an empty string.
- Do not copy the full prompt unless a short phrase is required.

Schema:
{
  "goal": "",
  "success": [],
  "must": [],
  "must_not": [],
  "exact": [],
  "assume": [],
  "unknown": [],
  "facts": [],
  "now": "",
  "next": [],
  "blocked": [],
  "done": [],
  "risks": []
}

Task:
$TaskPrompt
"@
}

function Extract-JsonObject {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $null
    }

    $start = $Text.IndexOf('{')
    $end = $Text.LastIndexOf('}')
    if ($start -lt 0 -or $end -lt $start) {
        return $null
    }

    return $Text.Substring($start, $end - $start + 1)
}

function Invoke-CardGeneration {
    param(
        [string]$BaseUrl,
        [string]$TaskPrompt,
        [string]$CardPromptMode,
        [int]$CardMaxTokens,
        [int]$PollSeconds,
        [switch]$NoCardLint
    )

    $body = @{
        task = "card_generation"
        prompt = (New-CardPrompt -TaskPrompt $TaskPrompt)
        prompt_mode = $CardPromptMode
        max_tokens = $CardMaxTokens
        temperature = 0.0
        top_p = 1.0
        repetition_penalty = 1.0
        stream = $false
        seed = 7777
    }

    $status = Submit-GenerationJob -BaseUrl $BaseUrl -Body $body -PollSeconds $PollSeconds
    $rawText = [string]$status.result.text
    $jsonText = Extract-JsonObject -Text $rawText

    $card = $null
    $lint = @()
    $parseError = $null

    if ($jsonText) {
        try {
            $card = $jsonText | ConvertFrom-Json -Depth 8
        } catch {
            $parseError = $_.Exception.Message
        }
    } else {
        $parseError = "No JSON object found in card generation output"
    }

    if (-not $NoCardLint -and $card) {
        if ([string]::IsNullOrWhiteSpace([string]$card.goal)) {
            $lint += "missing_goal"
        }
        if ($null -eq $card.now -or [string]::IsNullOrWhiteSpace([string]$card.now)) {
            $lint += "missing_now"
        }
        if ($null -eq $card.must_not) {
            $lint += "missing_must_not"
        }
        if ($null -eq $card.unknown) {
            $lint += "missing_unknown"
        }
        if ($null -eq $card.exact) {
            $lint += "missing_exact"
        }
    }

    [pscustomobject]@{
        status = $status
        raw_text = $rawText
        json_text = $jsonText
        card = $card
        parse_error = $parseError
        lint = $lint
    }
}

function New-InstructionPrompt {
    param([string]$TaskPrompt)

    @"
Before solving this task, internally create a compact task card with goal, must, must_not, exact, assume, unknown, now, next, done, and risks. Use that card as your operative state. Do not output the card unless necessary. Then solve the task.

Task:
$TaskPrompt
"@
}

function New-PrefixPrompt {
    param(
        [string]$CardJson,
        [string]$TaskPrompt
    )

    @"
TASK CARD (authoritative working state):
$CardJson

ORIGINAL TASK:
$TaskPrompt
"@
}

function New-BootstrapPrompt {
    param([string]$CardJson)

    @"
Use only the following task card as your operative state. Do not rely on omitted prior prompt text. If the card contains unknowns, respect them instead of inventing missing details.

TASK CARD:
$CardJson
"@
}

function Invoke-ModeRun {
    param(
        [string]$BaseUrl,
        [string]$Mode,
        [string]$TaskPrompt,
        [string]$TaskPromptMode,
        [string]$CardPromptMode,
        [int]$TaskMaxTokens,
        [int]$CardMaxTokens,
        [int]$PollSeconds,
        [switch]$NoCardLint
    )

    $cardResult = $null
    $effectivePrompt = $TaskPrompt
    $notes = @()

    switch ($Mode) {
        "off" {
            $notes += "baseline"
        }
        "instruction" {
            $effectivePrompt = New-InstructionPrompt -TaskPrompt $TaskPrompt
            $notes += "instruction_only"
        }
        "prefix" {
            $cardResult = Invoke-CardGeneration -BaseUrl $BaseUrl -TaskPrompt $TaskPrompt -CardPromptMode $CardPromptMode -CardMaxTokens $CardMaxTokens -PollSeconds $PollSeconds -NoCardLint:$NoCardLint
            if (-not $cardResult.json_text) {
                throw "Prefix mode could not obtain a usable card: $($cardResult.parse_error)"
            }
            $effectivePrompt = New-PrefixPrompt -CardJson $cardResult.json_text -TaskPrompt $TaskPrompt
            $notes += "card_prefix_plus_original_task"
        }
        "bootstrap" {
            $cardResult = Invoke-CardGeneration -BaseUrl $BaseUrl -TaskPrompt $TaskPrompt -CardPromptMode $CardPromptMode -CardMaxTokens $CardMaxTokens -PollSeconds $PollSeconds -NoCardLint:$NoCardLint
            if (-not $cardResult.json_text) {
                throw "Bootstrap mode could not obtain a usable card: $($cardResult.parse_error)"
            }
            $effectivePrompt = New-BootstrapPrompt -CardJson $cardResult.json_text
            $notes += "two_pass_card_bootstrap"
        }
        default {
            throw "Unknown mode: $Mode"
        }
    }

    $body = @{
        task = "card_fail_fast_$Mode"
        prompt = $effectivePrompt
        prompt_mode = $TaskPromptMode
        max_tokens = $TaskMaxTokens
        temperature = 0.0
        top_p = 1.0
        repetition_penalty = 1.0
        stream = $false
        seed = 7777
    }

    $status = Submit-GenerationJob -BaseUrl $BaseUrl -Body $body -PollSeconds $PollSeconds

    [pscustomobject]@{
        mode = $Mode
        notes = $notes
        effective_prompt = $effectivePrompt
        card_result = $cardResult
        status = $status
    }
}

$taskPrompt = Get-TaskPrompt
Wait-ServerReady -BaseUrl $BaseUrl

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $OutputRoot $stamp
$null = New-Item -ItemType Directory -Force -Path $runDir

Set-Content -Path (Join-Path $runDir "task_prompt.txt") -Value $taskPrompt -NoNewline

$results = @()
foreach ($mode in $Modes) {
    Write-Host "Running mode: $mode"
    $modeDir = Join-Path $runDir $mode
    $null = New-Item -ItemType Directory -Force -Path $modeDir

    try {
        $result = Invoke-ModeRun -BaseUrl $BaseUrl -Mode $mode -TaskPrompt $taskPrompt -TaskPromptMode $TaskPromptMode -CardPromptMode $CardPromptMode -TaskMaxTokens $TaskMaxTokens -CardMaxTokens $CardMaxTokens -PollSeconds $PollSeconds -NoCardLint:$NoCardLint
        $results += $result

        Set-Content -Path (Join-Path $modeDir "effective_prompt.txt") -Value ([string]$result.effective_prompt) -NoNewline
        Set-Content -Path (Join-Path $modeDir "response.txt") -Value ([string]$result.status.result.text) -NoNewline
        $result.status | ConvertTo-Json -Depth 10 | Set-Content -Path (Join-Path $modeDir "status.json")

        if ($result.card_result) {
            Set-Content -Path (Join-Path $modeDir "card_raw.txt") -Value ([string]$result.card_result.raw_text) -NoNewline
            if ($result.card_result.json_text) {
                Set-Content -Path (Join-Path $modeDir "card.json") -Value ([string]$result.card_result.json_text) -NoNewline
            }
            $result.card_result | ConvertTo-Json -Depth 10 | Set-Content -Path (Join-Path $modeDir "card_generation.json")
        }
    } catch {
        $errorRecord = [pscustomobject]@{
            mode = $mode
            error = $_.Exception.Message
        }
        $results += $errorRecord
        $errorRecord | ConvertTo-Json -Depth 10 | Set-Content -Path (Join-Path $modeDir "error.json")
        Set-Content -Path (Join-Path $modeDir "error.txt") -Value ([string]$_.Exception.Message) -NoNewline
    }
}

$summary = foreach ($result in $results) {
    if ($result.PSObject.Properties.Name -contains "status") {
        [pscustomobject]@{
            mode = $result.mode
            outcome = "completed"
            stop_reason = [string]$result.status.result.stop_reason
            tokens_generated = [int]$result.status.result.tokens_generated
            chars = ([string]$result.status.result.text).Length
            card_generated = [bool]($null -ne $result.card_result)
            card_parse_error = if ($result.card_result) { [string]$result.card_result.parse_error } else { $null }
            card_lint = if ($result.card_result) { @($result.card_result.lint) } else { @() }
            mode_notes = @($result.notes)
            error = $null
        }
    } else {
        [pscustomobject]@{
            mode = $result.mode
            outcome = "failed"
            stop_reason = $null
            tokens_generated = $null
            chars = $null
            card_generated = $false
            card_parse_error = $null
            card_lint = @()
            mode_notes = @()
            error = [string]$result.error
        }
    }
}

$summary | ConvertTo-Json -Depth 10 | Set-Content -Path (Join-Path $runDir "summary.json")
$summary | Format-Table -AutoSize | Out-String | Set-Content -Path (Join-Path $runDir "summary.txt")

Write-Host "Run directory: $runDir"
$summary | ConvertTo-Json -Depth 10